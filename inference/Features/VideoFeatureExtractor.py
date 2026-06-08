import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import os
import logging


class VideoFeatureExtractor():
    def __init__(self, FPS=2.0, device="cuda"):
        self.FPS = FPS
        self.device = device

        # ✅ 加载 ResNet152（等价 TF）
        model = models.resnet152(pretrained=True)

        # 去掉 fc → 保留 avgpool
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.eval().to(self.device)

    def _tf_preprocess(self, frame):
        """
        模拟 TF keras.applications.resnet.preprocess_input
        非常关键！！
        """
        frame = frame.astype(np.float32)

        # OpenCV 是 BGR → TF 默认也是BGR模式
        # 减 ImageNet mean（BGR顺序）
        frame[..., 0] -= 103.939
        frame[..., 1] -= 116.779
        frame[..., 2] -= 123.68

        return frame

    def _extract_frames(self, video_path):
        """
        等价 FrameCV(FPS=2)
        """
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_rate = int(round(fps / self.FPS))

        frames = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % sample_rate == 0:
                # resize → 模拟 SoccerNet crop
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)

            idx += 1

        cap.release()
        return np.array(frames)

    def extractFeatures(self, path_video_input, path_features_output, overwrite=False):

        if os.path.exists(path_features_output) and not overwrite:
            logging.info("Features already exists, skipping.")
            return

        logging.info(f"Extracting features from {path_video_input}")

        frames = self._extract_frames(path_video_input)

        logging.info(f"Frames shape: {frames.shape}")

        # ✅ TF风格 preprocess
        frames = np.array([self._tf_preprocess(f) for f in frames])

        # HWC → CHW
        frames = frames.transpose(0, 3, 1, 2)

        frames = torch.from_numpy(frames).float().to(self.device)

        features = []

        with torch.no_grad():
            for i in range(0, len(frames), 64):
                batch = frames[i:i+64]
                out = self.model(batch)  # (B, 2048, 1, 1)
                out = out.view(out.size(0), -1)  # (B, 2048)
                features.append(out.cpu().numpy())

        features = np.concatenate(features, axis=0)

        logging.info(f"Features shape: {features.shape}")

        os.makedirs(os.path.dirname(path_features_output), exist_ok=True)
        np.save(path_features_output, features)


class PCAReducer():
    def __init__(self, pca_file=None, scaler_file=None):
        import pickle as pkl

        self.pca = None
        self.average = None

        if pca_file:
            with open(pca_file, "rb") as f:
                self.pca = pkl.load(f)

        if scaler_file:
            with open(scaler_file, "rb") as f:
                self.average = pkl.load(f)

    def reduceFeatures(self, input_features, output_features, overwrite=False):
        if os.path.exists(output_features) and not overwrite:
            return

        feat = np.load(input_features)

        if self.average is not None:
            feat = feat - self.average

        if self.pca is not None:
            feat = self.pca.transform(feat)

        np.save(output_features, feat)