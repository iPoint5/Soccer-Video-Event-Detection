import os
import logging
import numpy as np
import torch

from dataset import SoccerNetClips, SoccerNetClipsTesting
from model import ContextAwareModel,TCNModel
from train import trainer, test
from loss import ContextAwareLoss, SpottingLoss


def build_datasets(cfg):
    if not cfg["test_only"]:
        dataset_train = SoccerNetClips(
            path=cfg["SoccerNet_path"],
            features=cfg["features"],
            split="train",
            framerate=cfg["framerate"],
            chunk_size=cfg["chunk_size"] * cfg["framerate"],
            receptive_field=cfg["receptive_field"] * cfg["framerate"],
            chunks_per_epoch=cfg["chunks_per_epoch"]
        )

        dataset_valid = SoccerNetClips(
            path=cfg["SoccerNet_path"],
            features=cfg["features"],
            split="valid",
            framerate=cfg["framerate"],
            chunk_size=cfg["chunk_size"] * cfg["framerate"],
            receptive_field=cfg["receptive_field"] * cfg["framerate"],
            chunks_per_epoch=cfg["chunks_per_epoch"]
        )

        dataset_valid_metric = SoccerNetClipsTesting(
            path=cfg["SoccerNet_path"],
            features=cfg["features"],
            split="valid",
            framerate=cfg["framerate"],
            chunk_size=cfg["chunk_size"] * cfg["framerate"],
            receptive_field=cfg["receptive_field"] * cfg["framerate"]
        )
    else:
        dataset_train = dataset_valid = dataset_valid_metric = None

    split_to_test = "challenge" if cfg["challenge"] else "test"

    dataset_test = SoccerNetClipsTesting(
        path=cfg["SoccerNet_path"],
        features=cfg["features"],
        split=split_to_test,
        framerate=cfg["framerate"],
        chunk_size=cfg["chunk_size"] * cfg["framerate"],
        receptive_field=cfg["receptive_field"] * cfg["framerate"]
    )

    return dataset_train, dataset_valid, dataset_valid_metric, dataset_test


def build_model(cfg, dataset_test):
    model = TCNModel(
        weights=cfg["load_weights"],
        input_size=cfg["num_features"],
        num_classes=dataset_test.num_classes,
        chunk_size=cfg["chunk_size"] * cfg["framerate"],
        dim_capsule=cfg["dim_capsule"],
        receptive_field=cfg["receptive_field"] * cfg["framerate"],
        num_detections=dataset_test.num_detections,
        framerate=cfg["framerate"]
    ).cuda()

    logging.info(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params}")

    return model


def build_dataloaders(cfg, dataset_train, dataset_valid, dataset_valid_metric, dataset_test):
    loaders = {}

    if not cfg["test_only"]:
        loaders["train"] = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True,
            persistent_workers=True
        )

        loaders["valid"] = torch.utils.data.DataLoader(
            dataset_valid,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=True,
            persistent_workers=True
        )

        loaders["valid_metric"] = torch.utils.data.DataLoader(
            dataset_valid_metric,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

    loaders["test"] = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    return loaders


def run_experiment(cfg):
    # 固定随机性
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    logging.info("==== CONFIG ====")
    for k, v in cfg.items():
        logging.info(f"{k}: {v}")

    # ===== 数据 =====
    dataset_train, dataset_valid, dataset_valid_metric, dataset_test = build_datasets(cfg)

    # ===== 模型 =====
    model = build_model(cfg, dataset_test)

    # ===== DataLoader =====
    loaders = build_dataloaders(cfg, dataset_train, dataset_valid, dataset_valid_metric, dataset_test)

    # ===== 训练 =====
    if not cfg["test_only"]:
        criterion_seg = ContextAwareLoss(K=dataset_train.K_parameters)
        criterion_spot = SpottingLoss(
            lambda_coord=cfg["lambda_coord"],
            lambda_noobj=cfg["lambda_noobj"]
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["LR"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.6,
            patience=2,
            threshold=1e-3,
            cooldown=1,
            min_lr=1e-6
        )
        # scheduler = None

        trainer(
            loaders["train"],
            loaders["valid"],
            loaders["valid_metric"],
            loaders["test"],
            model,
            optimizer,
            scheduler,
            [criterion_seg, criterion_spot],
            [cfg["loss_weight_seg"], cfg["loss_weight_det"]],
            model_name=cfg["model_name"],
            max_epochs=cfg["max_epochs"],
            evaluation_frequency=cfg["eval_freq"]
        )


if __name__ == "__main__":
        
    cfg = {
        "SoccerNet_path": "data/",
        "features": "ResNET_PCA512.npy",
        "framerate": 2,
        "chunk_size": 120,
        "receptive_field": 40,
        "chunks_per_epoch": 6000,
        "num_features": 512,
        "dim_capsule": 16,
        "batch_size": 32,
        "num_workers": 4,
        "LR": 5e-4,
        "patience": 10,
        "max_epochs": 300,
        "eval_freq": 5,
        "lambda_coord": 5,
        "lambda_noobj": 0.5,
        "loss_weight_seg": 0.000367,
        "loss_weight_det": 1.0,
        "model_name": "tcn_promax",
        "load_weights": None,
        "test_only": False,
        "challenge": False,
        "seed": 42
    }

    run_experiment(cfg)