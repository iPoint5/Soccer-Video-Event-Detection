
import __future__

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F




class ContextAwareModel(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=3, chunk_size=240, dim_capsule=16, receptive_field=80, num_detections=5, framerate=2):
        """
        INPUT: a Tensor of the form (batch_size,1,chunk_size,input_size)
        OUTPUTS:    1. The segmentation of the form (batch_size,chunk_size,num_classes)
                    2. The action spotting of the form (batch_size,num_detections,2+num_classes)
        """

        super(ContextAwareModel, self).__init__()

        self.load_weights(weights=weights)

        self.input_size = input_size
        self.num_classes = num_classes
        self.dim_capsule = dim_capsule
        self.receptive_field = receptive_field
        self.num_detections = num_detections
        self.chunk_size = chunk_size
        self.framerate = framerate

        self.pyramid_size_1 = int(np.ceil(receptive_field/7))
        self.pyramid_size_2 = int(np.ceil(receptive_field/3))
        self.pyramid_size_3 = int(np.ceil(receptive_field/2))
        self.pyramid_size_4 = int(np.ceil(receptive_field))

        # Base Convolutional Layers
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,input_size))
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1,1))

        # Temporal Pyramidal Module
        self.pad_p_1 = nn.ZeroPad2d((0,0,(self.pyramid_size_1-1)//2, self.pyramid_size_1-1-(self.pyramid_size_1-1)//2))
        self.pad_p_2 = nn.ZeroPad2d((0,0,(self.pyramid_size_2-1)//2, self.pyramid_size_2-1-(self.pyramid_size_2-1)//2))
        self.pad_p_3 = nn.ZeroPad2d((0,0,(self.pyramid_size_3-1)//2, self.pyramid_size_3-1-(self.pyramid_size_3-1)//2))
        self.pad_p_4 = nn.ZeroPad2d((0,0,(self.pyramid_size_4-1)//2, self.pyramid_size_4-1-(self.pyramid_size_4-1)//2))
        self.conv_p_1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(self.pyramid_size_1,1))
        self.conv_p_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.pyramid_size_2,1))
        self.conv_p_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(self.pyramid_size_3,1))
        self.conv_p_4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(self.pyramid_size_4,1))

        # -------------------
        # Segmentation module
        # -------------------

        self.kernel_seg_size = 3
        self.pad_seg = nn.ZeroPad2d((0,0,(self.kernel_seg_size-1)//2, self.kernel_seg_size-1-(self.kernel_seg_size-1)//2))
        self.conv_seg = nn.Conv2d(in_channels=152, out_channels=dim_capsule*num_classes, kernel_size=(self.kernel_seg_size,1))
        self.batch_seg = nn.BatchNorm2d(num_features=self.chunk_size, momentum=0.01,eps=0.001) 


        # -------------------
        # detection module
        # -------------------       
        self.max_pool_spot = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        self.kernel_spot_size = 3
        self.pad_spot_1 = nn.ZeroPad2d((0,0,(self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        self.conv_spot_1 = nn.Conv2d(in_channels=num_classes*(dim_capsule+1), out_channels=32, kernel_size=(self.kernel_spot_size,1))
        self.max_pool_spot_1 = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        self.pad_spot_2 = nn.ZeroPad2d((0,0,(self.kernel_spot_size-1)//2, self.kernel_spot_size-1-(self.kernel_spot_size-1)//2))
        self.conv_spot_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(self.kernel_spot_size,1))
        self.max_pool_spot_2 = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))

        # Confidence branch
        self.conv_conf = nn.Conv2d(in_channels=16*(chunk_size//8-1), out_channels=self.num_detections*2, kernel_size=(1,1))

        # Class branch
        self.conv_class = nn.Conv2d(in_channels=16*(chunk_size//8-1), out_channels=self.num_detections*self.num_classes, kernel_size=(1,1))
        self.softmax = nn.Softmax(dim=-1)


    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, inputs):

        # -----------------------------------
        # Feature input (chunks of the video)
        # -----------------------------------
        # input_shape: (batch,channel,frames,dim_features)
        #print("Input size: ", inputs.size())

        # -------------------------------------
        # Temporal Convolutional neural network
        # -------------------------------------


        # Base Convolutional Layers
        conv_1 = F.relu(self.conv_1(inputs))
        #print("Conv_1 size: ", conv_1.size())
        
        conv_2 = F.relu(self.conv_2(conv_1))
        #print("Conv_2 size: ", conv_2.size())


        # Temporal Pyramidal Module
        conv_p_1 = F.relu(self.conv_p_1(self.pad_p_1(conv_2)))
        #print("Conv_p_1 size: ", conv_p_1.size())
        conv_p_2 = F.relu(self.conv_p_2(self.pad_p_2(conv_2)))
        #print("Conv_p_2 size: ", conv_p_2.size())
        conv_p_3 = F.relu(self.conv_p_3(self.pad_p_3(conv_2)))
        #print("Conv_p_3 size: ", conv_p_3.size())
        conv_p_4 = F.relu(self.conv_p_4(self.pad_p_4(conv_2)))
        #print("Conv_p_4 size: ", conv_p_4.size())

        concatenation = torch.cat((conv_2,conv_p_1,conv_p_2,conv_p_3,conv_p_4),1)
        #print("Concatenation size: ", concatenation.size())


        # -------------------
        # Segmentation module
        # -------------------

        conv_seg = self.conv_seg(self.pad_seg(concatenation))
        #print("Conv_seg size: ", conv_seg.size())

        conv_seg_permuted = conv_seg.permute(0,2,3,1)
        #print("Conv_seg_permuted size: ", conv_seg_permuted.size())

        conv_seg_reshaped = conv_seg_permuted.view(conv_seg_permuted.size()[0],conv_seg_permuted.size()[1],self.dim_capsule,self.num_classes)
        #print("Conv_seg_reshaped size: ", conv_seg_reshaped.size())


        #conv_seg_reshaped_permuted = conv_seg_reshaped.permute(0,3,1,2)
        #print("Conv_seg_reshaped_permuted size: ", conv_seg_reshaped_permuted.size())

        conv_seg_norm = torch.sigmoid(self.batch_seg(conv_seg_reshaped))
        #print("Conv_seg_norm: ", conv_seg_norm.size())


        #conv_seg_norm_permuted = conv_seg_norm.permute(0,2,3,1)
        #print("Conv_seg_norm_permuted size: ", conv_seg_norm_permuted.size())

        output_segmentation = torch.sqrt(torch.sum(torch.square(conv_seg_norm-0.5), dim=2)*4/self.dim_capsule)
        #print("Output_segmentation size: ", output_segmentation.size())


        # ---------------
        # Spotting module
        # ---------------

        # Concatenation of the segmentation score to the capsules
        output_segmentation_reverse = 1-output_segmentation
        #print("Output_segmentation_reverse size: ", output_segmentation_reverse.size())

        output_segmentation_reverse_reshaped = output_segmentation_reverse.unsqueeze(2)
        #print("Output_segmentation_reverse_reshaped size: ", output_segmentation_reverse_reshaped.size())


        output_segmentation_reverse_reshaped_permutted = output_segmentation_reverse_reshaped.permute(0,3,1,2)
        #print("Output_segmentation_reverse_reshaped_permutted size: ", output_segmentation_reverse_reshaped_permutted.size())

        concatenation_2 = torch.cat((conv_seg, output_segmentation_reverse_reshaped_permutted), dim=1)
        #print("Concatenation_2 size: ", concatenation_2.size())

        conv_spot = self.max_pool_spot(F.relu(concatenation_2))
        #print("Conv_spot size: ", conv_spot.size())

        conv_spot_1 = F.relu(self.conv_spot_1(self.pad_spot_1(conv_spot)))
        #print("Conv_spot_1 size: ", conv_spot_1.size())

        conv_spot_1_pooled = self.max_pool_spot_1(conv_spot_1)
        #print("Conv_spot_1_pooled size: ", conv_spot_1_pooled.size())

        conv_spot_2 = F.relu(self.conv_spot_2(self.pad_spot_2(conv_spot_1_pooled)))
        #print("Conv_spot_2 size: ", conv_spot_2.size())

        conv_spot_2_pooled = self.max_pool_spot_2(conv_spot_2)
        #print("Conv_spot_2_pooled size: ", conv_spot_2_pooled.size())

        spotting_reshaped = conv_spot_2_pooled.view(conv_spot_2_pooled.size()[0],-1,1,1)
        #print("Spotting_reshape size: ", spotting_reshaped.size())

        # Confindence branch
        conf_pred = torch.sigmoid(self.conv_conf(spotting_reshaped).view(spotting_reshaped.shape[0],self.num_detections,2))
        #print("Conf_pred size: ", conf_pred.size())

        # Class branch
        conf_class = self.softmax(self.conv_class(spotting_reshaped).view(spotting_reshaped.shape[0],self.num_detections,self.num_classes))
        #print("Conf_class size: ", conf_class.size())

        output_spotting = torch.cat((conf_pred,conf_class),dim=-1)
        #print("Output_spotting size: ", output_spotting.size())


        return output_segmentation, output_spotting
        

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ContextAwareModel(nn.Module):
#     def __init__(self, weights=None, input_size=512, num_classes=3,
#                  chunk_size=240, dim_capsule=16, receptive_field=80,
#                  num_detections=5, framerate=2):

#         super(ContextAwareModel, self).__init__()
#         self.framerate = framerate
#         self.input_size = input_size
#         self.num_classes = num_classes
#         self.dim_capsule = dim_capsule
#         self.receptive_field = receptive_field
#         self.num_detections = num_detections
#         self.chunk_size = chunk_size

#         # ======================
#         # Base Layers
#         # ======================
#         self.conv_1 = nn.Conv2d(1, 128, (1, input_size))
#         self.conv_2 = nn.Conv2d(128, 32, (1, 1))

#         # ======================
#         # Pyramid
#         # ======================
#         def pad(k):
#             return nn.ZeroPad2d((0, 0, (k-1)//2, k-1-(k-1)//2))

#         self.pyramid_size_1 = int(np.ceil(receptive_field/7))
#         self.pyramid_size_2 = int(np.ceil(receptive_field/3))
#         self.pyramid_size_3 = int(np.ceil(receptive_field/2))
#         self.pyramid_size_4 = int(np.ceil(receptive_field))

#         self.pad_p_1 = pad(self.pyramid_size_1)
#         self.pad_p_2 = pad(self.pyramid_size_2)
#         self.pad_p_3 = pad(self.pyramid_size_3)
#         self.pad_p_4 = pad(self.pyramid_size_4)

#         self.conv_p_1 = nn.Conv2d(32, 8, (self.pyramid_size_1, 1))
#         self.conv_p_2 = nn.Conv2d(32, 16, (self.pyramid_size_2, 1))
#         self.conv_p_3 = nn.Conv2d(32, 32, (self.pyramid_size_3, 1))
#         self.conv_p_4 = nn.Conv2d(32, 64, (self.pyramid_size_4, 1))

#         # ======================
#         # Segmentation
#         # ======================
#         self.pad_seg = pad(3)
#         self.conv_seg = nn.Conv2d(152, dim_capsule*num_classes, (3, 1))

#         # ======================
#         # Spotting (NEW: Transformer)
#         # ======================
#         self.max_pool_spot = nn.MaxPool2d((3,1), stride=(2,1))
#         self.conv_spot_1 = nn.Conv2d(num_classes*(dim_capsule+1), 32, (3,1), padding=(1,0))
#         self.conv_spot_2 = nn.Conv2d(32, 16, (3,1), padding=(1,0))

#         self.max_pool_spot_1 = nn.MaxPool2d((3,1), stride=(2,1))
#         self.max_pool_spot_2 = nn.MaxPool2d((3,1), stride=(2,1))

#         # ===== Transformer =====
#         self.d_model = 16
#         max_T = chunk_size // 8

#         self.pos_embedding = nn.Parameter(torch.randn(1, max_T, self.d_model))

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.d_model,
#             nhead=4,
#             dim_feedforward=64,
#             dropout=0.1,
#             batch_first=True
#         )

#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

#         # ===== Detection heads（已修正）=====
#         self.conv_conf = nn.Conv2d(16, num_detections*2, (1,1))
#         self.conv_class = nn.Conv2d(16, num_detections*num_classes, (1,1))

#         self.softmax = nn.Softmax(dim=-1)

#         if weights is not None:
#             self.load_weights(weights)

#     def load_weights(self, weights):
#         checkpoint = torch.load(weights)
#         self.load_state_dict(checkpoint['state_dict'])

#     def forward(self, inputs):

#         # ======================
#         # Base
#         # ======================
#         x = F.relu(self.conv_1(inputs))
#         x = F.relu(self.conv_2(x))

#         # ======================
#         # Pyramid
#         # ======================
#         p1 = F.relu(self.conv_p_1(self.pad_p_1(x)))
#         p2 = F.relu(self.conv_p_2(self.pad_p_2(x)))
#         p3 = F.relu(self.conv_p_3(self.pad_p_3(x)))
#         p4 = F.relu(self.conv_p_4(self.pad_p_4(x)))

#         x = torch.cat((x, p1, p2, p3, p4), dim=1)  # (B,152,T,1)

#         # ======================
#         # Segmentation
#         # ======================
#         seg = self.conv_seg(self.pad_seg(x))  # (B,C,T,1)

#         B, C, T, _ = seg.shape
#         seg = seg.view(B, self.dim_capsule, self.num_classes, T)
#         seg = seg.permute(0, 3, 1, 2)  # (B,T,dim_capsule,num_classes)

#         seg = torch.sigmoid(seg)

#         output_segmentation = torch.sqrt(
#             torch.sum((seg - 0.5)**2, dim=2) * 4 / self.dim_capsule
#         )

#         # ======================
#         # Spotting
#         # ======================
#         reverse = 1 - output_segmentation
#         reverse = reverse.unsqueeze(2).permute(0,3,1,2)

#         seg_raw = self.conv_seg(self.pad_seg(x))  # raw feature
#         x = torch.cat((seg_raw, reverse), dim=1)

#         x = self.max_pool_spot(F.relu(x))
#         x = F.relu(self.conv_spot_1(x))
#         x = self.max_pool_spot_1(x)

#         x = F.relu(self.conv_spot_2(x))
#         x = self.max_pool_spot_2(x)  # (B,16,T',1)

#         # ===== Transformer =====
#         x = x.squeeze(-1)           # (B,16,T)
#         x = x.permute(0,2,1)        # (B,T,16)

#         T_cur = x.shape[1]
#         x = x + self.pos_embedding[:, :T_cur, :]

#         x = self.transformer(x)     # (B,T,16)

#         x = x.permute(0,2,1).unsqueeze(-1)  # (B,16,T,1)

#         # ===== Pooling → Detection =====
#         k = self.num_detections

#         x_flat = x.squeeze(-1)  # (B,C,T)

#         x_topk, _ = torch.topk(x_flat, k=k, dim=2)

#         x_pool = x_topk.mean(dim=2, keepdim=True).unsqueeze(-1)

#         conf = torch.sigmoid(
#             self.conv_conf(x_pool).view(B, self.num_detections, 2)*2
#         )

#         cls = self.softmax(
#             self.conv_class(x_pool).view(B, self.num_detections, self.num_classes)
#         )

#         output_spotting = torch.cat((conf, cls), dim=-1)

#         return output_segmentation, output_spotting


# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # =========================
# # TCN Block (时序卷积块)
# # =========================
# class TCNBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         # 使用 Same Padding 确保时序长度 (L) 不变
#         # padding = dilation * (kernel_size - 1) / 2
#         self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, dilation=1)
#         self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2)
#         self.conv3 = nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4)
#         self.bn = nn.BatchNorm1d(channels)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         residual = x
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.bn(x)
#         return x + residual

# # =========================
# # Main Model
# # =========================
# class ContextAwareModel(nn.Module):
#     def __init__(
#         self,
#         weights=None,
#         input_size=512,
#         num_classes=3,
#         chunk_size=240,
#         dim_capsule=16,
#         receptive_field=80,
#         num_detections=5,
#         framerate=2,
#     ):
#         super(ContextAwareModel, self).__init__()

#         self.input_size = input_size
#         self.num_classes = num_classes
#         self.dim_capsule = dim_capsule
#         self.receptive_field = receptive_field
#         self.num_detections = num_detections
#         self.chunk_size = chunk_size
#         self.framerate = framerate

#         # 1. Base Conv 层
#         self.conv_1 = nn.Conv2d(1, 128, kernel_size=(1, input_size))
#         self.conv_2 = nn.Conv2d(128, 32, kernel_size=(1, 1))

#         # 2. 时序金字塔模块参数
#         self.pyramid_size_1 = int(np.ceil(receptive_field / 7))
#         self.pyramid_size_2 = int(np.ceil(receptive_field / 3))
#         self.pyramid_size_3 = int(np.ceil(receptive_field / 2))
#         self.pyramid_size_4 = int(np.ceil(receptive_field))

#         self.pad_p_1 = nn.ZeroPad2d((0, 0, (self.pyramid_size_1 - 1)//2, self.pyramid_size_1 - 1 - (self.pyramid_size_1 - 1)//2))
#         self.pad_p_2 = nn.ZeroPad2d((0, 0, (self.pyramid_size_2 - 1)//2, self.pyramid_size_2 - 1 - (self.pyramid_size_2 - 1)//2))
#         self.pad_p_3 = nn.ZeroPad2d((0, 0, (self.pyramid_size_3 - 1)//2, self.pyramid_size_3 - 1 - (self.pyramid_size_3 - 1)//2))
#         self.pad_p_4 = nn.ZeroPad2d((0, 0, (self.pyramid_size_4 - 1)//2, self.pyramid_size_4 - 1 - (self.pyramid_size_4 - 1)//2))

#         self.conv_p_1 = nn.Conv2d(32, 8, kernel_size=(self.pyramid_size_1, 1))
#         self.conv_p_2 = nn.Conv2d(32, 16, kernel_size=(self.pyramid_size_2, 1))
#         self.conv_p_3 = nn.Conv2d(32, 32, kernel_size=(self.pyramid_size_3, 1))
#         self.conv_p_4 = nn.Conv2d(32, 64, kernel_size=(self.pyramid_size_4, 1))

#         # 3. ⭐ TCN 增强模块 (放在特征融合后)
#         # 总通道数 = 32 (conv_2) + 8 + 16 + 32 + 64 = 152
#         self.tcn_module = nn.Sequential(
#             TCNBlock(152),
#             TCNBlock(152)
#         )

#         # 4. Segmentation 模块
#         self.conv_seg = nn.Conv2d(152, dim_capsule * num_classes, kernel_size=(3, 1), padding=(1, 0))
#         self.batch_seg = nn.BatchNorm2d(dim_capsule * num_classes)

#         # 5. Spotting 模块
#         self.max_pool_spot = nn.MaxPool2d((3, 1), stride=(2, 1))
#         self.conv_spot_1 = nn.Conv2d(num_classes * (dim_capsule + 1), 32, kernel_size=(3, 1), padding=(1, 0))
#         self.max_pool_spot_1 = nn.MaxPool2d((3, 1), stride=(2, 1))
#         self.conv_spot_2 = nn.Conv2d(32, 16, kernel_size=(3, 1), padding=(1, 0))
#         self.max_pool_spot_2 = nn.MaxPool2d((3, 1), stride=(2, 1))

#         # 经过三次 max_pool (3,1) stride 2，240 -> 119 -> 59 -> 29
#         # 原公式：chunk_size // 8 - 1 = 240 // 8 - 1 = 29
#         self.feature_map_len = 29 
#         self.conv_conf = nn.Conv2d(16 * self.feature_map_len, num_detections * 2, kernel_size=(1, 1))
#         self.conv_class = nn.Conv2d(16 * self.feature_map_len, num_detections * num_classes, kernel_size=(1, 1))

#         self.softmax = nn.Softmax(dim=-1)
#         self.load_weights(weights)

#     def forward(self, inputs):
#         batch_size = inputs.size(0)

#         # --- 特征提取 ---
#         x = F.relu(self.conv_1(inputs))
#         x = F.relu(self.conv_2(x))

#         # --- 金字塔多尺度融合 ---
#         p1 = F.relu(self.conv_p_1(self.pad_p_1(x)))
#         p2 = F.relu(self.conv_p_2(self.pad_p_2(x)))
#         p3 = F.relu(self.conv_p_3(self.pad_p_3(x)))
#         p4 = F.relu(self.conv_p_4(self.pad_p_4(x)))
#         x_concat = torch.cat((x, p1, p2, p3, p4), dim=1) # (B, 152, T, 1)

#         # --- ⭐ TCN 时序建模 ---
#         # 转换到 1D 卷积格式 (B, C, T)
#         tcn_in = x_concat.squeeze(-1) 
#         tcn_out = self.tcn_module(tcn_in)
#         x_enhanced = tcn_out.unsqueeze(-1) # 回到 (B, 152, T, 1)

#         # --- Segmentation 分支 ---
#         conv_seg = self.conv_seg(x_enhanced) 
#         conv_seg = self.batch_seg(conv_seg)
#         conv_seg = torch.sigmoid(conv_seg) # (B, dim_cap*num_cls, T, 1)

#         # 转换为 Capsule 形状: (B, T, dim_cap, num_cls)
#         conv_seg_permuted = conv_seg.squeeze(-1).permute(0, 2, 1) 
#         conv_seg_reshaped = conv_seg_permuted.view(batch_size, self.chunk_size, self.dim_capsule, self.num_classes)

#         # 计算分割概率 (B, T, num_cls)
#         output_seg = torch.sqrt(torch.sum((conv_seg_reshaped - 0.5) ** 2, dim=2) * 4 / self.dim_capsule)

#         # --- Spotting 分支 ---
#         # 1. 准备反向概率引导: (B, T, num_cls) -> (B, num_cls, T, 1)
#         rev_seg = (1 - output_seg).permute(0, 2, 1).unsqueeze(-1)

#         # 2. 准备 Capsule 特征: (B, dim_cap*num_cls, T, 1)
#         # 将 capsule 按类别分组，方便后续操作
#         # 原逻辑：cat(capsules, 1-prob)
#         # 为了匹配原模型，我们需要保证通道顺序
#         x_spot = torch.cat((conv_seg, rev_seg), dim=1) # (B, num_cls*dim_cap + num_cls, T, 1)

#         # 3. 卷积与池化下采样
#         x_spot = self.max_pool_spot(F.relu(x_spot))
#         x_spot = F.relu(self.conv_spot_1(x_spot))
#         x_spot = self.max_pool_spot_1(x_spot)
#         x_spot = F.relu(self.conv_spot_2(x_spot))
#         x_spot = self.max_pool_spot_2(x_spot)

#         # 4. 展平并预测
#         x_spot = x_spot.view(batch_size, -1, 1, 1)
        
#         conf = torch.sigmoid(self.conv_conf(x_spot).view(batch_size, self.num_detections, 2))
#         cls = self.softmax(self.conv_class(x_spot).view(batch_size, self.num_detections, self.num_classes))
        
#         output_spot = torch.cat((conf, cls), dim=-1)

#         return output_seg, output_spot

#     def load_weights(self, weights=None):
#         if weights is not None:
#             print(f"=> Loading {weights}")
#             checkpoint = torch.load(weights)
#             self.load_state_dict(checkpoint["state_dict"], strict=False)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 基础组件 (保留你的 TCN 设计)
# =========================
class DilatedTCNBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3, 
            padding=dilation, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.relu(self.bn(self.conv(x)))
        return x + res

class TCNStack(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            DilatedTCNBlock(channels, dilation=1),
            DilatedTCNBlock(channels, dilation=2),
            DilatedTCNBlock(channels, dilation=4),
            DilatedTCNBlock(channels, dilation=8),
        )
    def forward(self, x):
        return self.layers(x)

class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, 1, kernel_size=1)
    def forward(self, x):
        attn = torch.sigmoid(self.conv(x))
        return x * attn

# =========================
# 完整修复后的模型
# =========================
class TCNModel(nn.Module):
    def __init__(
        self,
        weights=None,
        input_size=512,
        num_classes=3,
        chunk_size=240,
        dim_capsule=16,
        receptive_field=80,
        num_detections=5,
        framerate=2,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.dim_capsule = dim_capsule
        self.receptive_field = receptive_field
        self.num_detections = num_detections
        self.chunk_size = chunk_size
        self.framerate = framerate

        # 1. 基础卷积 (修复 AttributeError)
        self.conv_1 = nn.Conv2d(1, 128, kernel_size=(1, input_size))
        self.conv_2 = nn.Conv2d(128, 32, kernel_size=(1, 1))

        # 2. 金字塔配置
        self.pyramid_size_1 = int(np.ceil(receptive_field / 7))
        self.pyramid_size_2 = int(np.ceil(receptive_field / 3))
        self.pyramid_size_3 = int(np.ceil(receptive_field / 2))
        self.pyramid_size_4 = int(np.ceil(receptive_field))

        def get_pad(k):
            return nn.ZeroPad2d((0, 0, (k - 1)//2, k - 1 - (k - 1)//2))

        self.pad_p1 = get_pad(self.pyramid_size_1)
        self.pad_p2 = get_pad(self.pyramid_size_2)
        self.pad_p3 = get_pad(self.pyramid_size_3)
        self.pad_p4 = get_pad(self.pyramid_size_4)

        self.conv_p1 = nn.Conv2d(32, 8, kernel_size=(self.pyramid_size_1, 1))
        self.conv_p2 = nn.Conv2d(32, 16, kernel_size=(self.pyramid_size_2, 1))
        self.conv_p3 = nn.Conv2d(32, 32, kernel_size=(self.pyramid_size_3, 1))
        self.conv_p4 = nn.Conv2d(32, 64, kernel_size=(self.pyramid_size_4, 1))

        # 3. ⭐ 并行 TCN (给每个分支)
        self.tcn_x = TCNStack(32)
        self.tcn_p1 = TCNStack(8)
        self.tcn_p2 = TCNStack(16)
        self.tcn_p3 = TCNStack(32)
        self.tcn_p4 = TCNStack(64)

        # 4. 融合与全局 TCN
        total_channels = 32 + 8 + 16 + 32 + 64 # 152
        self.global_tcn = TCNStack(total_channels)
        self.temporal_attn = TemporalAttention(total_channels)

        # 5. 分割头
        self.conv_seg = nn.Conv2d(total_channels, dim_capsule * num_classes, kernel_size=(3,1), padding=(1,0))
        self.bn_seg = nn.BatchNorm2d(dim_capsule * num_classes)

        # 6. ⭐ 修复后的 Spotting 头 (加入下采样防止维度爆炸)
        # 输入维度: capsule特征 (dim_cap * num_cls) + 引导分值 (num_cls)
        spot_in_channels = num_classes * (dim_capsule + 1)
        self.spot_downsample = nn.Sequential(
            nn.Conv1d(spot_in_channels, 64, kernel_size=3, stride=2, padding=1), # 240 -> 120
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1), # 120 -> 60
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1), # 60 -> 30
            nn.ReLU(),
        )
        
        # 此时 T = 30. 我们将特征展平
        self.conf_head = nn.Conv1d(32 * 30, num_detections * 2, kernel_size=1)
        self.cls_head = nn.Conv1d(32 * 30, num_detections * num_classes, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)

        if weights:
            self.load_weights(weights)

    def forward(self, inputs):
        B = inputs.size(0)

        # --- 基础特征 ---
        x = F.relu(self.conv_1(inputs))
        x = F.relu(self.conv_2(x)) # (B, 32, T, 1)

        # --- 并行 TCN 分支 ---
        # 每一个分支都要 squeeze 进入 TCNBlock (1D)，再 unsqueeze 回来
        x_t = self.tcn_x(x.squeeze(-1)).unsqueeze(-1)
        
        p1 = self.tcn_p1(F.relu(self.conv_p1(self.pad_p1(x))).squeeze(-1)).unsqueeze(-1)
        p2 = self.tcn_p2(F.relu(self.conv_p2(self.pad_p2(x))).squeeze(-1)).unsqueeze(-1)
        p3 = self.tcn_p3(F.relu(self.conv_p3(self.pad_p3(x))).squeeze(-1)).unsqueeze(-1)
        p4 = self.tcn_p4(F.relu(self.conv_p4(self.pad_p4(x))).squeeze(-1)).unsqueeze(-1)

        # --- 融合 ---
        x_cat = torch.cat([x_t, p1, p2, p3, p4], dim=1).squeeze(-1) # (B, 152, T)

        # --- 全局时序建模 ---
        x_global = self.global_tcn(x_cat)
        x_global = self.temporal_attn(x_global) # (B, 152, T)
#==============================================
        # --- 分割分支 ---
        seg_2d = x_global.unsqueeze(-1)
        # 注意：这里的 sigmoid 确保了 conv_seg 在 [0, 1]
        conv_seg = torch.sigmoid(self.bn_seg(self.conv_seg(seg_2d))) 

        # 计算分割图 (B, T, num_cls)
        conv_seg_p = conv_seg.squeeze(-1).permute(0, 2, 1)
        conv_seg_r = conv_seg_p.view(B, self.chunk_size, self.dim_capsule, self.num_classes)
        
        # ✅ 修复点 1：使用更小的 eps，并放在 sqrt 外部或进行有效截断
        dist_sq = torch.sum((conv_seg_r - 0.5) ** 2, dim=2) * 4 / self.dim_capsule
        
        # ✅ 修复点 2：极其重要！必须限制在 1.0 以内
        # 我们先开方，然后用 clamp 强制限制在 [1e-7, 1-1e-7] 之间
        # 这样既保护了 log(x) 也不保护了 log(1-x)
        output_segmentation = torch.sqrt(dist_sq + 1e-9) 
        output_segmentation = torch.clamp(output_segmentation, min=1e-7, max=1.0 - 1e-7)
#==========================================
        # --- Spotting 分支 ---
        # 1. 拼接反向引导分值
        rev_seg = (1 - output_segmentation).permute(0, 2, 1) # (B, num_cls, T)
        # spot_input: (B, num_cls*dim_cap + num_cls, T)
        spot_input = torch.cat([conv_seg.squeeze(-1), rev_seg], dim=1)

        # 2. 下采样时序维度到 30
        spot_feat = self.spot_downsample(spot_input) # (B, 32, 30)
        
        # 3. 展平所有时序特征用于检测头
        spot_flat = spot_feat.view(B, -1, 1) # (B, 32*30, 1)
        
        # 4. 检测预测
        conf_pred = torch.sigmoid(self.conf_head(spot_flat).view(B, self.num_detections, 2))
        class_pred = self.softmax(self.cls_head(spot_flat).view(B, self.num_detections, self.num_classes))
        
        output_spotting = torch.cat([conf_pred, class_pred], dim=-1)

        return output_segmentation, output_spotting

    def load_weights(self, weights=None):
        if weights:
            print(f"=> loading checkpoint '{weights}'")
            checkpoint = torch.load(weights, map_location='cpu')
            # 必须设置 strict=False 因为结构发生了重大改变
            self.load_state_dict(checkpoint['state_dict'], strict=False)