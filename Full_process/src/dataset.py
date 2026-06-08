from torch.utils.data import Dataset

import numpy as np
import random
# import pandas as pd
import os
import time


from tqdm import tqdm
# import utils

import torch

import logging
import json

from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from config.classes import EVENT_DICTIONARY_V2, K_V2

from preprocessing import oneHotToShifts, getTimestampTargets, getChunks_anchors

import random




class SoccerNetClips(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split="train",
                 framerate=2, chunk_size=240, receptive_field=80,
                 chunks_per_epoch=6000, seed=42):

        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.chunks_per_epoch = chunks_per_epoch
        self.split = split

        # ✅ 固定随机种子（可复现）
        random.seed(seed)
        np.random.seed(seed)

        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = 17
        self.labels = "Labels-v2.json"
        self.K_parameters = K_V2 * framerate
        self.num_detections = 15

        logging.info("Checking/Downloading data")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(
            files=[self.labels, f"1_{self.features}", f"2_{self.features}"],
            split=[split], verbose=False
        )

        # ========= 数据容器 =========
        self.game_feats = []
        self.game_labels = []
        self.game_anchors = [[] for _ in range(self.num_classes + 1)]

        logging.info("Pre-compute clips")

        game_counter = 0

        for game in tqdm(self.listGames):

            # ===== load features（建议后续改成 mmap）=====
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))

            labels = json.load(open(os.path.join(self.path, game, self.labels)))

            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))

            # ===== 构建 one-hot 标签 =====
            for ann in labels["annotations"]:
                time = ann["gameTime"]
                event = ann["label"]

                if event not in self.dict_event:
                    continue

                half = int(time[0])
                minutes = int(time[-5:-3])
                seconds = int(time[-2:])
                frame = framerate * (seconds + 60 * minutes)

                label = self.dict_event[event]

                if half == 1:
                    frame = min(frame, feat_half1.shape[0] - 1)
                    label_half1[frame][label] = 1
                else:
                    frame = min(frame, feat_half2.shape[0] - 1)
                    label_half2[frame][label] = 1

            # ===== shift =====
            shift_half1 = oneHotToShifts(label_half1, self.K_parameters)
            shift_half2 = oneHotToShifts(label_half2, self.K_parameters)

            # ===== anchors =====
            anchors_half1 = getChunks_anchors(
                shift_half1, game_counter,
                self.K_parameters, self.chunk_size, self.receptive_field
            )
            game_counter += 1

            anchors_half2 = getChunks_anchors(
                shift_half2, game_counter,
                self.K_parameters, self.chunk_size, self.receptive_field
            )
            game_counter += 1

            # ===== 存储 =====
            self.game_feats.extend([feat_half1, feat_half2])
            self.game_labels.extend([shift_half1, shift_half2])

            for anchor in anchors_half1:
                self.game_anchors[anchor[2]].append(anchor)
            for anchor in anchors_half2:
                self.game_anchors[anchor[2]].append(anchor)

    def __getitem__(self, index):

        # ========= 类别采样（修复不平衡） =========
        if random.random() < 0.5:
            class_selection = self.num_classes  # background
        else:
            class_selection = random.randint(0, self.num_classes - 1)

        anchors = self.game_anchors[class_selection]
        anchor_info = anchors[random.randint(0, len(anchors) - 1)]

        game_index = anchor_info[0]
        anchor = anchor_info[1]

        # ========= shift策略（更均匀） =========
        if class_selection < self.num_classes:
            shift = np.random.randint(-self.chunk_size // 2, self.chunk_size // 2)
            start = anchor + shift
        else:
            start = random.randint(anchor[0], anchor[1] - self.chunk_size)

        # ========= 边界安全 =========
        max_start = self.game_feats[game_index].shape[0] - self.chunk_size
        start = max(0, min(start, max_start))

        # ========= clip抽取 =========
        clip_feat = self.game_feats[game_index][start:start + self.chunk_size]

        # ❗关键修复：必须 copy
        clip_labels = self.game_labels[game_index][start:start + self.chunk_size].copy()

        # ========= receptive field mask =========
        pad = int(np.ceil(self.receptive_field / 2))
        clip_labels[:pad] = -1
        clip_labels[-pad:] = -1

        # ========= detection target =========
        clip_targets = getTimestampTargets(
            np.array([clip_labels]), self.num_detections
        )[0]

        return (
            torch.from_numpy(clip_feat).float(),
            torch.from_numpy(clip_labels).float(),
            torch.from_numpy(clip_targets).float()
        )

    def __len__(self):
        return self.chunks_per_epoch

class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", split="test", 
                framerate=2, chunk_size=240, receptive_field=80):
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.framerate = framerate

        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = 17
        self.labels="Labels-v2.json"
        self.K_parameters = K_V2*framerate
        self.num_detections =15
        self.split=split

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)
        if split == "challenge":
            downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False)
        else:       
            downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[split], verbose=False)



    def __getitem__(self, index):
        
        # Load features
        feat_half1 = np.load(os.path.join(self.path, self.listGames[index], "1_" + self.features))
        feat_half2 = np.load(os.path.join(self.path, self.listGames[index], "2_" + self.features))


        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))


        # Load labels
        if os.path.exists(os.path.join(self.path, self.listGames[index], self.labels)):
            labels = json.load(open(os.path.join(self.path, self.listGames[index], self.labels)))

            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = self.framerate * ( seconds + 60 * minutes ) 

                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]

                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_half1[frame][label] = value

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_half2[frame][label] = value

        def feats2clip(feats, stride, clip_length):

            idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
            idxs = []
            for i in torch.arange(0, clip_length):
                idxs.append(idx+i)
            idx = torch.stack(idxs, dim=1)

            idx = idx.clamp(0, feats.shape[0]-1)
            idx[-1] = torch.arange(clip_length)+feats.shape[0]-clip_length

            return feats[idx,:]
            

        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=self.chunk_size-self.receptive_field, 
                        clip_length=self.chunk_size)

        feat_half2 = feats2clip(torch.from_numpy(feat_half2), 
                        stride=self.chunk_size-self.receptive_field, 
                        clip_length=self.chunk_size)
                                  
        return feat_half1, feat_half2, torch.from_numpy(label_half1), torch.from_numpy(label_half2)

    def __len__(self):
        return len(self.listGames)