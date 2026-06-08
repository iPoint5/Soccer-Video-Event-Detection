import logging
import os
from metrics_visibility_fast import AverageMeter, average_mAP, NMS
import time
from tqdm import tqdm
import torch
import numpy as np
import math
from preprocessing import batch2long, timestamps2long
from json_io import predictions2json
from SoccerNet.Downloader import getListGames

import csv


def trainer(train_loader,
            val_loader,
            val_metric_loader,
            test_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            weights,
            model_name,
            max_epochs=1000,
            evaluation_frequency=20):
    best_loss = 9e99
    best_metric = -1

    # 🔥 延迟启用 scheduler
    scheduler_enabled = False

    # 记录
    loss_history = []
    map_history = []

    for epoch in range(max_epochs):

        # =========================
        # Train
        # =========================
        loss_training = train(
            train_loader,
            model,
            criterion,
            weights,
            optimizer,
            epoch + 1,
            train=True)

        # =========================
        # Validation Loss
        # =========================
        loss_validation = train(
            val_loader,
            model,
            criterion,
            weights,
            optimizer,
            epoch + 1,
            train=False)

        # 记录 loss
        loss_history.append({
            "epoch": epoch + 1,
            "train_loss": float(loss_training),
            "val_loss": float(loss_validation)
        })

        if epoch % evaluation_frequency == 0 and epoch != 0:

            results = test(
                val_metric_loader,
                model,
                model_name)

            (a_mAP,
             a_mAP_per_class,
             a_mAP_visible,
             a_mAP_per_class_visible,
             a_mAP_unshown,
             a_mAP_per_class_unshown) = results

            logging.info(f"[Epoch {epoch+1}] mAP: {a_mAP:.4f}")

            # =========================
            # 🔥 激活 scheduler 条件
            # =========================
            if (not scheduler_enabled) and (a_mAP > 0.38):
                scheduler_enabled = True
                logging.info(
                    f"Scheduler ACTIVATED at epoch {epoch+1}, mAP={a_mAP:.4f}")

            # =========================
            # 🔥 调用 scheduler（基于 mAP）
            # =========================
            if scheduler_enabled:
                prevLR = optimizer.param_groups[0]['lr']

                scheduler.step(a_mAP)

                currLR = optimizer.param_groups[0]['lr']

                if currLR != prevLR:
                    logging.info(
                        f"LR reduced: {prevLR:.6f} -> {currLR:.6f}")

                # 🔥 可选 early stop（防止无效训练）
                if currLR < 1e-6 and scheduler.num_bad_epochs >= scheduler.patience:
                    logging.info(
                        "Early stopping: LR too small & no improvement")
                    break

            # =========================
            # 记录 mAP
            # =========================
            record = {
                "epoch": epoch + 1,
                "a_mAP": float(a_mAP),
                "a_mAP_visible": float(a_mAP_visible),
                "a_mAP_unshown": float(a_mAP_unshown),
            }

            for i, v in enumerate(a_mAP_per_class):
                record[f"class_{i}_mAP"] = float(v)

            map_history.append(record)

            best_metric = max(a_mAP, best_metric)

    # =========================
    # 保存 CSV
    # =========================
    save_dir = os.path.join("models", model_name)
    os.makedirs(save_dir, exist_ok=True)

    # -------- loss.csv --------
    if len(loss_history) > 0:
        loss_path = os.path.join(save_dir, "loss.csv")
        keys = loss_history[0].keys()

        with open(loss_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(loss_history)

        logging.info(f"Loss saved to {loss_path}")

    # -------- map.csv --------
    if len(map_history) > 0:
        map_path = os.path.join(save_dir, "map.csv")
        keys = map_history[0].keys()

        with open(map_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(map_history)

        logging.info(f"mAP saved to {map_path}")

    final_model_path = os.path.join(save_dir, "final_model200.pth.tar")

    state = {
        'epoch': max_epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, final_model_path)

    logging.info(f"Final model saved to {final_model_path}")

    return

def train(dataloader,
          model,
          criterion, 
          weights,
          optimizer,
          epoch,
          train=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_segmentation = AverageMeter()
    losses_spotting = AverageMeter()

    # switch to train / eval mode
    if train:
        model.train()
    else:
        model.eval()
        
    end = time.time()

    with tqdm(enumerate(dataloader),
              total=len(dataloader),
              ncols=120,
              mininterval=0.5) as t:   # ✅ 限制刷新频率

        for i, (feats, labels, targets) in t: 

            # ===== 数据加载时间 =====
            data_time.update(time.time() - end)

            # ===== 数据搬到GPU =====
            feats = feats.cuda()
            labels = labels.cuda().float()
            targets = targets.cuda().float()

            feats = feats.unsqueeze(1)

            # ===== 前向 =====
            output_segmentation, output_spotting = model(feats)

            loss_segmentation = criterion[0](labels, output_segmentation) 
            loss_spotting = criterion[1](targets, output_spotting)

            loss = weights[0] * loss_segmentation + weights[1] * loss_spotting

            # ===== 统计 =====
            losses.update(loss.item(), feats.size(0))
            losses_segmentation.update(loss_segmentation.item(), feats.size(0))
            losses_spotting.update(loss_spotting.item(), feats.size(0))

            # ===== 反向传播 =====
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # ===== 时间统计 =====
            batch_time.update(time.time() - end)
            end = time.time()

            # ===== tqdm 显示（核心优化）=====
            if train:
                t.set_description(f"Train {epoch}")
            else:
                t.set_description(f"Eval {epoch}")

            # 每10步更新一次，避免刷屏 + 提高性能
            if i % 10 == 0:
                t.set_postfix({
                    "loss": f"{losses.avg:.4f}",
                    "seg": f"{losses_segmentation.avg:.4f}",
                    "spot": f"{losses_spotting.avg:.4f}",
                    "dt": f"{data_time.avg:.3f}s",
                    "bt": f"{batch_time.avg:.3f}s"
                })

    return losses.avg


def test(dataloader,model, model_name, save_predictions=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    spotting_grountruth = list()
    spotting_grountruth_visibility = list()
    spotting_predictions = list()
    segmentation_predictions = list()

    chunk_size = model.chunk_size
    receptive_field = model.receptive_field

    model.eval()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        for i, (feat_half1, feat_half2, label_half1, label_half2) in t:
            data_time.update(time.time() - end)

            feat_half1 = feat_half1.cuda().squeeze(0)
            label_half1 = label_half1.float().squeeze(0)
            feat_half2 = feat_half2.cuda().squeeze(0)
            label_half2 = label_half2.float().squeeze(0)


            feat_half1=feat_half1.unsqueeze(1)
            feat_half2=feat_half2.unsqueeze(1)

            # Compute the output
            output_segmentation_half_1, output_spotting_half_1 = model(feat_half1)
            output_segmentation_half_2, output_spotting_half_2 = model(feat_half2)


            timestamp_long_half_1 = timestamps2long(output_spotting_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
            timestamp_long_half_2 = timestamps2long(output_spotting_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)
            segmentation_long_half_1 = batch2long(output_segmentation_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
            segmentation_long_half_2 = batch2long(output_segmentation_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)

            spotting_grountruth.append(torch.abs(label_half1))
            spotting_grountruth.append(torch.abs(label_half2))
            spotting_grountruth_visibility.append(label_half1)
            spotting_grountruth_visibility.append(label_half2)
            spotting_predictions.append(timestamp_long_half_1)
            spotting_predictions.append(timestamp_long_half_2)
            segmentation_predictions.append(segmentation_long_half_1)
            segmentation_predictions.append(segmentation_long_half_2)


    # Transformation to numpy for evaluation
    targets_numpy = list()
    closests_numpy = list()
    detections_numpy = list()
    for target, detection in zip(spotting_grountruth_visibility,spotting_predictions):
        target_numpy = target.numpy()
        targets_numpy.append(target_numpy)
        detections_numpy.append(NMS(detection.numpy(), 20*model.framerate))
        
        closest_numpy = np.zeros(target_numpy.shape)-1
        #Get the closest action index
        for c in np.arange(target_numpy.shape[-1]):
            indexes = np.where(target_numpy[:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = target_numpy[indexes[i],c]
        closests_numpy.append(closest_numpy)

    # Save the predictions to the json format
    if save_predictions:
        list_game = getListGames(dataloader.dataset.split)
        for index in np.arange(len(list_game)):
            predictions2json(detections_numpy[index*2], detections_numpy[(index*2)+1],"outputs/", list_game[index], model.framerate)


    # Compute the performances
    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = average_mAP(targets_numpy, detections_numpy, closests_numpy, model.framerate)
    
    print("Average mAP: ", a_mAP)
    print("Average mAP visible: ", a_mAP_visible)
    print("Average mAP unshown: ", a_mAP_unshown)
    # print("Average mAP per class: ", a_mAP_per_class)
    # print("Average mAP visible per class: ", a_mAP_per_class_visible)
    # print("Average mAP unshown per class: ", a_mAP_per_class_unshown)

    return a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown