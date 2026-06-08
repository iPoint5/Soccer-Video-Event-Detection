from torch.utils.data import Dataset

import numpy as np
import random
# import pandas as pd
import os
import time
import ffmpy

from tqdm import tqdm
# import utils

import torch

import logging
import json

from SoccerNet.Downloader import SoccerNetDownloader
from Features.VideoFeatureExtractor import VideoFeatureExtractor, PCAReducer

# class SoccerNetClipsTesting(Dataset):
#     def __init__(self, path, features="ResNET_PCA512.npy", 
#                 framerate=2, chunk_size=240, receptive_field=80):
#         self.path = path
#         self.chunk_size = chunk_size
#         self.receptive_field = receptive_field
#         self.framerate = framerate
#         self.num_classes = 17
#         self.num_detections =15

#         #Changing video format to 
#         ff = ffmpy.FFmpeg(
#              inputs={self.path: ""},
#              outputs={"inference/outputs/videoLQ.mkv": '-y -r 25 -vf scale=-1:224 -max_muxing_queue_size 9999'})
#         print(ff.cmd)
#         ff.run()

#         print("Initializing feature extractor")
#         myFeatureExtractor = VideoFeatureExtractor(
#             feature="ResNET",
#             back_end="TF2",
#             transform="crop",
#             grabber="opencv",
#             FPS=self.framerate)

#         print("Extracting frames")
#         myFeatureExtractor.extractFeatures(path_video_input="inference/outputs/videoLQ.mkv",
#                                            path_features_output="inference/outputs/features.npy",
#                                            overwrite=True)

#         print("Initializing PCA reducer")
#         myPCAReducer = PCAReducer(pca_file="inference/Features/pca_512_TF2.pkl",
#                                   scaler_file="inference/Features/average_512_TF2.pkl")

#         print("Reducing with PCA")
#         myPCAReducer.reduceFeatures(input_features="inference/outputs/features.npy",
#                                     output_features="inference/outputs/features_PCA.npy",
#                                     overwrite=True)



#     def __getitem__(self, index):



        
#         # Load features
#         feat_half1 = np.load(os.path.join("inference/outputs/features_PCA.npy"))
#         print("Shape half 1: ", feat_half1.shape)
#         size = feat_half1.shape[0]

#         def feats2clip(feats, stride, clip_length):

#             idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
#             idxs = []
#             for i in torch.arange(0, clip_length):
#                 idxs.append(idx+i)
#             idx = torch.stack(idxs, dim=1)

#             idx = idx.clamp(0, feats.shape[0]-1)
#             idx[-1] = torch.arange(clip_length)+feats.shape[0]-clip_length

#             return feats[idx,:]
            

#         feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
#                         stride=self.chunk_size-self.receptive_field, 
#                         clip_length=self.chunk_size)
                                  
#         return feat_half1, size

#     def __len__(self):
#         return 1
    

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
        if isinstance(K_V2, torch.Tensor):
            self.K_parameters = (K_V2 * framerate).cpu().numpy()
        else:
            self.K_parameters = K_V2 * framerate
        # ✅ 可复现
        random.seed(seed)
        np.random.seed(seed)

        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = 17
        self.labels = "Labels-v2.json"
        self.num_detections = 15

        logging.info("Checking/Downloading data")
        downloader = SoccerNetDownloader(path)
        downloader.downloadGames(
            files=[self.labels, f"1_{self.features}", f"2_{self.features}"],
            split=[split], verbose=False
        )

        # ========= 只存路径（关键！省内存） =========
        self.game_feat_paths = []
        self.game_labels = []
        self.game_anchors = [[] for _ in range(self.num_classes + 1)]

        logging.info("Pre-compute anchors")

        game_counter = 0

        for game in tqdm(self.listGames):

            feat_path1 = os.path.join(self.path, game, "1_" + self.features)
            feat_path2 = os.path.join(self.path, game, "2_" + self.features)

            # ⚠️ 只读取 shape（不会加载整个数组）
            feat_half1 = np.load(feat_path1, mmap_mode='r')
            feat_half2 = np.load(feat_path2, mmap_mode='r')

            labels = json.load(open(os.path.join(self.path, game, self.labels)))

            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))

            # ===== 构建标签 =====
            for ann in labels["annotations"]:
                event = ann["label"]
                if event not in self.dict_event:
                    continue

                time = ann["gameTime"]
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

            # ===== 存储路径（不存特征！）=====
            self.game_feat_paths.extend([feat_path1, feat_path2])
            self.game_labels.extend([shift_half1, shift_half2])

            for anchor in anchors_half1:
                self.game_anchors[anchor[2]].append(anchor)
            for anchor in anchors_half2:
                self.game_anchors[anchor[2]].append(anchor)

    def __getitem__(self, index):

        # ========= 类别采样（平衡） =========
        class_selection = random.randint(0, self.num_classes)

        anchors = self.game_anchors[class_selection]
        anchor_info = anchors[random.randint(0, len(anchors) - 1)]

        game_index = anchor_info[0]
        anchor = anchor_info[1]

        # ========= shift（均匀） =========
        if class_selection < self.num_classes:
            shift = np.random.randint(-self.chunk_size // 2, self.chunk_size // 2)
            start = anchor + shift
        else:
            start = random.randint(anchor[0], anchor[1] - self.chunk_size)

        # ========= mmap加载（核心省内存） =========
        feat = np.load(self.game_feat_paths[game_index], mmap_mode='r')

        # ========= 边界安全 =========
        max_start = feat.shape[0] - self.chunk_size
        start = max(0, min(start, max_start))

        # ========= 切片（不会复制数据） =========
        clip_feat = feat[start:start + self.chunk_size]

        # ========= label（必须copy） =========
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
            torch.from_numpy(clip_feat).float(),      # ✅ float32 防止显存翻倍
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
        self.labels = "Labels-v2.json"
        self.num_detections = 15
        self.split = split

        # ✅ 修复：保证是 numpy
        if isinstance(K_V2, torch.Tensor):
            self.K_parameters = (K_V2 * framerate).cpu().numpy()
        else:
            self.K_parameters = K_V2 * framerate

        logging.info("Checking/Downloading data")
        downloader = SoccerNetDownloader(path)

        if split == "challenge":
            downloader.downloadGames(
                files=[f"1_{self.features}", f"2_{self.features}"],
                split=[split], verbose=False)
        else:
            downloader.downloadGames(
                files=[self.labels, f"1_{self.features}", f"2_{self.features}"],
                split=[split], verbose=False)

        # ✅ 只存路径（关键优化）
        self.game_feat_paths = []
        self.game_labels = []

        for game in self.listGames:

            feat_path1 = os.path.join(self.path, game, "1_" + self.features)
            feat_path2 = os.path.join(self.path, game, "2_" + self.features)

            feat_half1 = np.load(feat_path1, mmap_mode='r')
            feat_half2 = np.load(feat_path2, mmap_mode='r')

            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))

            label_path = os.path.join(self.path, game, self.labels)

            if os.path.exists(label_path):
                labels = json.load(open(label_path))

                for ann in labels["annotations"]:
                    event = ann["label"]
                    if event not in self.dict_event:
                        continue

                    time = ann["gameTime"]
                    half = int(time[0])
                    minutes = int(time[-5:-3])
                    seconds = int(time[-2:])
                    frame = self.framerate * (seconds + 60 * minutes)

                    label = self.dict_event[event]

                    value = 1
                    if "visibility" in ann and ann["visibility"] == "not shown":
                        value = -1

                    if half == 1:
                        frame = min(frame, feat_half1.shape[0] - 1)
                        label_half1[frame][label] = value
                    else:
                        frame = min(frame, feat_half2.shape[0] - 1)
                        label_half2[frame][label] = value

            # ✅ 只存路径 + label
            self.game_feat_paths.append((feat_path1, feat_path2))
            self.game_labels.append((label_half1, label_half2))

    def feats2clip_numpy(self, feats, stride, clip_length):
        """
        numpy版本，避免torch开销
        """
        num_frames = feats.shape[0]

        idx = np.arange(0, num_frames - 1, stride)
        idxs = [idx + i for i in range(clip_length)]
        idx = np.stack(idxs, axis=1)

        idx = np.clip(idx, 0, num_frames - 1)
        idx[-1] = np.arange(clip_length) + num_frames - clip_length

        return feats[idx]

    def __getitem__(self, index):

        feat_path1, feat_path2 = self.game_feat_paths[index]
        label_half1, label_half2 = self.game_labels[index]

        # ✅ mmap读取（不会占内存）
        feat_half1 = np.load(feat_path1, mmap_mode='r')
        feat_half2 = np.load(feat_path2, mmap_mode='r')

        # ✅ numpy完成切片
        feat_half1 = self.feats2clip_numpy(
            feat_half1,
            stride=self.chunk_size - self.receptive_field,
            clip_length=self.chunk_size
        )

        feat_half2 = self.feats2clip_numpy(
            feat_half2,
            stride=self.chunk_size - self.receptive_field,
            clip_length=self.chunk_size
        )

        # ✅ 最后才转 torch（关键）
        return (
            torch.from_numpy(feat_half1).float(),
            torch.from_numpy(feat_half2).float(),
            torch.from_numpy(label_half1).float(),
            torch.from_numpy(label_half2).float()
        )

    def __len__(self):
        return len(self.listGames)