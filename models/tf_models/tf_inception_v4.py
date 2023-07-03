import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_weights_dict = dict()


def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict


class IncV4KitModel(nn.Module):

    def __init__(self, weight_file):
        super(IncV4KitModel, self).__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)

        self.InceptionV4_InceptionV4_Conv2d_1a_3x3_Conv2D = self.__conv(2,
                                                                        name='InceptionV4/InceptionV4/Conv2d_1a_3x3/Conv2D',
                                                                        in_channels=3, out_channels=32,
                                                                        kernel_size=(3, 3), stride=(2, 2), groups=1,
                                                                        bias=None)
        self.InceptionV4_InceptionV4_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                         'InceptionV4/InceptionV4/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm',
                                                                                                         num_features=32,
                                                                                                         eps=0.0010000000474974513,
                                                                                                         momentum=0.0)
        self.InceptionV4_InceptionV4_Conv2d_2a_3x3_Conv2D = self.__conv(2,
                                                                        name='InceptionV4/InceptionV4/Conv2d_2a_3x3/Conv2D',
                                                                        in_channels=32, out_channels=32,
                                                                        kernel_size=(3, 3), stride=(1, 1), groups=1,
                                                                        bias=None)
        self.InceptionV4_InceptionV4_Conv2d_2a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                         'InceptionV4/InceptionV4/Conv2d_2a_3x3/BatchNorm/FusedBatchNorm',
                                                                                                         num_features=32,
                                                                                                         eps=0.0010000000474974513,
                                                                                                         momentum=0.0)
        self.InceptionV4_InceptionV4_Conv2d_2b_3x3_Conv2D = self.__conv(2,
                                                                        name='InceptionV4/InceptionV4/Conv2d_2b_3x3/Conv2D',
                                                                        in_channels=32, out_channels=64,
                                                                        kernel_size=(3, 3), stride=(1, 1), groups=1,
                                                                        bias=None)
        self.InceptionV4_InceptionV4_Conv2d_2b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                         'InceptionV4/InceptionV4/Conv2d_2b_3x3/BatchNorm/FusedBatchNorm',
                                                                                                         num_features=64,
                                                                                                         eps=0.0010000000474974513,
                                                                                                         momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_3a_Branch_1_Conv2d_0a_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_3a/Branch_1/Conv2d_0a_3x3/Conv2D',
                                                                                          in_channels=64,
                                                                                          out_channels=96,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(2, 2), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_3a_Branch_1_Conv2d_0a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_3a/Branch_1/Conv2d_0a_3x3/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_4a/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=160,
                                                                                          out_channels=64,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_4a/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=160,
                                                                                          out_channels=64,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_4a/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=64,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_4a/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=64,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_1a_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_4a/Branch_0/Conv2d_1a_3x3/Conv2D',
                                                                                          in_channels=64,
                                                                                          out_channels=96,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_4a/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                          in_channels=64,
                                                                                          out_channels=64,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_4a/Branch_0/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_4a/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm', num_features=64,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_4a/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                          in_channels=64,
                                                                                          out_channels=64,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_4a/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm', num_features=64,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_1a_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_4a/Branch_1/Conv2d_1a_3x3/Conv2D',
                                                                                          in_channels=64,
                                                                                          out_channels=96,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_4a/Branch_1/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5a_Branch_0_Conv2d_1a_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5a/Branch_0/Conv2d_1a_3x3/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=192,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(2, 2), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5a/Branch_0/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5b_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5b/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=96,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5b/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=64,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5b/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=64,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5b_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5b/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5b/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=64,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5b/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=64,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5b/Branch_3/Conv2d_0b_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=96,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5b/Branch_3/Conv2d_0b_1x1/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5b/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                          in_channels=64,
                                                                                          out_channels=96,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5b/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                          in_channels=64,
                                                                                          out_channels=96,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5b/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5b/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0c_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5b/Branch_2/Conv2d_0c_3x3/Conv2D',
                                                                                          in_channels=96,
                                                                                          out_channels=96,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5b/Branch_2/Conv2d_0c_3x3/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5c_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5c/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=96,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5c/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=64,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5c/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=64,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5c_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5c/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5c/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=64,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5c/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=64,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5c_Branch_3_Conv2d_0b_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5c/Branch_3/Conv2d_0b_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=96,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5c_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5c/Branch_3/Conv2d_0b_1x1/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5c/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                          in_channels=64,
                                                                                          out_channels=96,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5c/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                          in_channels=64,
                                                                                          out_channels=96,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5c/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5c/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0c_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5c/Branch_2/Conv2d_0c_3x3/Conv2D',
                                                                                          in_channels=96,
                                                                                          out_channels=96,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5c/Branch_2/Conv2d_0c_3x3/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5d_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5d/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=96,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5d/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=64,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5d/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=64,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5d_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5d/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5d/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=64,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5d/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=64,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5d_Branch_3_Conv2d_0b_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5d/Branch_3/Conv2d_0b_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=96,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5d_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5d/Branch_3/Conv2d_0b_1x1/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5d/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                          in_channels=64,
                                                                                          out_channels=96,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5d/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                          in_channels=64,
                                                                                          out_channels=96,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5d/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5d/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0c_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5d/Branch_2/Conv2d_0c_3x3/Conv2D',
                                                                                          in_channels=96,
                                                                                          out_channels=96,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5d/Branch_2/Conv2d_0c_3x3/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5e_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5e/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=96,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5e/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=64,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5e/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=64,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5e_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5e/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5e/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=64,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5e/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=64,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5e_Branch_3_Conv2d_0b_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5e/Branch_3/Conv2d_0b_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=96,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5e_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5e/Branch_3/Conv2d_0b_1x1/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5e/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                          in_channels=64,
                                                                                          out_channels=96,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5e/Branch_2/Conv2d_0b_3x3/Conv2D',
                                                                                          in_channels=64,
                                                                                          out_channels=96,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5e/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5e/Branch_2/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0c_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_5e/Branch_2/Conv2d_0c_3x3/Conv2D',
                                                                                          in_channels=96,
                                                                                          out_channels=96,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_5e/Branch_2/Conv2d_0c_3x3/BatchNorm/FusedBatchNorm', num_features=96,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6a_Branch_0_Conv2d_1a_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6a/Branch_0/Conv2d_1a_3x3/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=384,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(2, 2), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6a/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6a/Branch_0/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm', num_features=384,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6a/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0b_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6a/Branch_1/Conv2d_0b_3x3/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=224,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6a/Branch_1/Conv2d_0b_3x3/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_1a_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6a/Branch_1/Conv2d_1a_3x3/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=256,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(2, 2), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6a/Branch_1/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6b/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=384,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6b/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6b/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6b/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=384,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6b/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6b/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_3_Conv2d_0b_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6b/Branch_3/Conv2d_0b_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=128,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6b/Branch_3/Conv2d_0b_1x1/BatchNorm/FusedBatchNorm', num_features=128,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6b/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=224,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0b_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6b/Branch_2/Conv2d_0b_7x1/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=192,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6b/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6b/Branch_2/Conv2d_0b_7x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6b/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=256,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0c_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6b/Branch_2/Conv2d_0c_1x7/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=224,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6b/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6b/Branch_2/Conv2d_0c_1x7/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0d_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6b/Branch_2/Conv2d_0d_7x1/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=224,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6b/Branch_2/Conv2d_0d_7x1/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0e_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6b/Branch_2/Conv2d_0e_1x7/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6b/Branch_2/Conv2d_0e_1x7/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6c/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=384,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6c/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6c/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6c/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=384,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6c/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6c/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_3_Conv2d_0b_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6c/Branch_3/Conv2d_0b_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=128,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6c/Branch_3/Conv2d_0b_1x1/BatchNorm/FusedBatchNorm', num_features=128,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6c/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=224,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0b_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6c/Branch_2/Conv2d_0b_7x1/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=192,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6c/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6c/Branch_2/Conv2d_0b_7x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6c/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=256,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0c_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6c/Branch_2/Conv2d_0c_1x7/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=224,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6c/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6c/Branch_2/Conv2d_0c_1x7/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0d_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6c/Branch_2/Conv2d_0d_7x1/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=224,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6c/Branch_2/Conv2d_0d_7x1/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0e_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6c/Branch_2/Conv2d_0e_1x7/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6c/Branch_2/Conv2d_0e_1x7/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6d/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=384,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6d/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6d/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6d/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=384,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6d/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6d/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_3_Conv2d_0b_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6d/Branch_3/Conv2d_0b_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=128,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6d/Branch_3/Conv2d_0b_1x1/BatchNorm/FusedBatchNorm', num_features=128,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6d/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=224,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0b_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6d/Branch_2/Conv2d_0b_7x1/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=192,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6d/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6d/Branch_2/Conv2d_0b_7x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6d/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=256,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0c_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6d/Branch_2/Conv2d_0c_1x7/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=224,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6d/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6d/Branch_2/Conv2d_0c_1x7/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0d_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6d/Branch_2/Conv2d_0d_7x1/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=224,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6d/Branch_2/Conv2d_0d_7x1/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0e_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6d/Branch_2/Conv2d_0e_1x7/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6d/Branch_2/Conv2d_0e_1x7/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6e/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=384,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6e/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6e/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6e/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=384,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6e/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6e/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_3_Conv2d_0b_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6e/Branch_3/Conv2d_0b_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=128,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6e/Branch_3/Conv2d_0b_1x1/BatchNorm/FusedBatchNorm', num_features=128,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6e/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=224,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0b_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6e/Branch_2/Conv2d_0b_7x1/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=192,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6e/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6e/Branch_2/Conv2d_0b_7x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6e/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=256,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0c_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6e/Branch_2/Conv2d_0c_1x7/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=224,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6e/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6e/Branch_2/Conv2d_0c_1x7/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0d_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6e/Branch_2/Conv2d_0d_7x1/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=224,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6e/Branch_2/Conv2d_0d_7x1/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0e_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6e/Branch_2/Conv2d_0e_1x7/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6e/Branch_2/Conv2d_0e_1x7/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6f/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=384,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6f/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6f/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6f/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=384,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6f/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6f/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_3_Conv2d_0b_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6f/Branch_3/Conv2d_0b_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=128,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6f/Branch_3/Conv2d_0b_1x1/BatchNorm/FusedBatchNorm', num_features=128,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6f/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=224,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0b_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6f/Branch_2/Conv2d_0b_7x1/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=192,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6f/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6f/Branch_2/Conv2d_0b_7x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6f/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=256,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0c_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6f/Branch_2/Conv2d_0c_1x7/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=224,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6f/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6f/Branch_2/Conv2d_0c_1x7/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0d_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6f/Branch_2/Conv2d_0d_7x1/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=224,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6f/Branch_2/Conv2d_0d_7x1/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0e_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6f/Branch_2/Conv2d_0e_1x7/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6f/Branch_2/Conv2d_0e_1x7/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6g/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=384,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6g/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6g/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6g/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=384,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6g/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6g/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_3_Conv2d_0b_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6g/Branch_3/Conv2d_0b_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=128,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6g/Branch_3/Conv2d_0b_1x1/BatchNorm/FusedBatchNorm', num_features=128,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6g/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=224,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0b_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6g/Branch_2/Conv2d_0b_7x1/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=192,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6g/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6g/Branch_2/Conv2d_0b_7x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6g/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=256,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0c_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6g/Branch_2/Conv2d_0c_1x7/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=224,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6g/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6g/Branch_2/Conv2d_0c_1x7/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0d_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6g/Branch_2/Conv2d_0d_7x1/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=224,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6g/Branch_2/Conv2d_0d_7x1/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0e_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6g/Branch_2/Conv2d_0e_1x7/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6g/Branch_2/Conv2d_0e_1x7/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6h/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=384,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6h/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6h/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6h/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=384,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6h/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6h/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_3_Conv2d_0b_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6h/Branch_3/Conv2d_0b_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=128,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6h/Branch_3/Conv2d_0b_1x1/BatchNorm/FusedBatchNorm', num_features=128,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6h/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=224,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0b_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6h/Branch_2/Conv2d_0b_7x1/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=192,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6h/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6h/Branch_2/Conv2d_0b_7x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6h/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=256,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0c_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6h/Branch_2/Conv2d_0c_1x7/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=224,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6h/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6h/Branch_2/Conv2d_0c_1x7/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0d_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6h/Branch_2/Conv2d_0d_7x1/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=224,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6h/Branch_2/Conv2d_0d_7x1/BatchNorm/FusedBatchNorm', num_features=224,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0e_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_6h/Branch_2/Conv2d_0e_1x7/Conv2D',
                                                                                          in_channels=224,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_6h/Branch_2/Conv2d_0e_1x7/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7a/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=192,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7a/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1024,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7a/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7a/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_AuxLogits_Conv2d_1b_1x1_Conv2D = self.__conv(2,
                                                                      name='InceptionV4/AuxLogits/Conv2d_1b_1x1/Conv2D',
                                                                      in_channels=1024, out_channels=128,
                                                                      kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                                      bias=None)
        self.InceptionV4_AuxLogits_Conv2d_1b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                       'InceptionV4/AuxLogits/Conv2d_1b_1x1/BatchNorm/FusedBatchNorm',
                                                                                                       num_features=128,
                                                                                                       eps=0.0010000000474974513,
                                                                                                       momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_1a_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7a/Branch_0/Conv2d_1a_3x3/Conv2D',
                                                                                          in_channels=192,
                                                                                          out_channels=192,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(2, 2), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0b_1x7_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7a/Branch_1/Conv2d_0b_1x7/Conv2D',
                                                                                          in_channels=256,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 7),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7a/Branch_0/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm', num_features=192,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7a/Branch_1/Conv2d_0b_1x7/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_AuxLogits_Conv2d_2a_Conv2D = self.__conv(2, name='InceptionV4/AuxLogits/Conv2d_2a/Conv2D',
                                                                  in_channels=128, out_channels=768, kernel_size=(5, 5),
                                                                  stride=(1, 1), groups=1, bias=None)
        self.InceptionV4_AuxLogits_Conv2d_2a_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                   'InceptionV4/AuxLogits/Conv2d_2a/BatchNorm/FusedBatchNorm',
                                                                                                   num_features=768,
                                                                                                   eps=0.0010000000474974513,
                                                                                                   momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0c_7x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7a/Branch_1/Conv2d_0c_7x1/Conv2D',
                                                                                          in_channels=256,
                                                                                          out_channels=320,
                                                                                          kernel_size=(7, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7a/Branch_1/Conv2d_0c_7x1/BatchNorm/FusedBatchNorm', num_features=320,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_AuxLogits_Aux_logits_MatMul = self.__dense(name='InceptionV4/AuxLogits/Aux_logits/MatMul',
                                                                    in_features=768, out_features=1001, bias=True)
        self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_1a_3x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7a/Branch_1/Conv2d_1a_3x3/Conv2D',
                                                                                          in_channels=320,
                                                                                          out_channels=320,
                                                                                          kernel_size=(3, 3),
                                                                                          stride=(2, 2), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7a/Branch_1/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm', num_features=320,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7b/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1536,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7b/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1536,
                                                                                          out_channels=384,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7b/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1536,
                                                                                          out_channels=384,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7b/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7b/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=384,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7b/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=384,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_3_Conv2d_0b_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7b/Branch_3/Conv2d_0b_1x1/Conv2D',
                                                                                          in_channels=1536,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7b/Branch_3/Conv2d_0b_1x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0b_1x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7b/Branch_1/Conv2d_0b_1x3/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0c_3x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7b/Branch_1/Conv2d_0c_3x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=256,
                                                                                          kernel_size=(3, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0b_3x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7b/Branch_2/Conv2d_0b_3x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=448,
                                                                                          kernel_size=(3, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7b/Branch_1/Conv2d_0b_1x3/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7b/Branch_1/Conv2d_0c_3x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0b_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7b/Branch_2/Conv2d_0b_3x1/BatchNorm/FusedBatchNorm', num_features=448,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0c_1x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7b/Branch_2/Conv2d_0c_1x3/Conv2D',
                                                                                          in_channels=448,
                                                                                          out_channels=512,
                                                                                          kernel_size=(1, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0c_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7b/Branch_2/Conv2d_0c_1x3/BatchNorm/FusedBatchNorm', num_features=512,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0d_1x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7b/Branch_2/Conv2d_0d_1x3/Conv2D',
                                                                                          in_channels=512,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0e_3x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7b/Branch_2/Conv2d_0e_3x1/Conv2D',
                                                                                          in_channels=512,
                                                                                          out_channels=256,
                                                                                          kernel_size=(3, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0d_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7b/Branch_2/Conv2d_0d_1x3/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0e_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7b/Branch_2/Conv2d_0e_3x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7c/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1536,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7c/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1536,
                                                                                          out_channels=384,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7c/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1536,
                                                                                          out_channels=384,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7c/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7c/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=384,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7c/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=384,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_3_Conv2d_0b_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7c/Branch_3/Conv2d_0b_1x1/Conv2D',
                                                                                          in_channels=1536,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7c/Branch_3/Conv2d_0b_1x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0b_1x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7c/Branch_1/Conv2d_0b_1x3/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0c_3x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7c/Branch_1/Conv2d_0c_3x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=256,
                                                                                          kernel_size=(3, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0b_3x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7c/Branch_2/Conv2d_0b_3x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=448,
                                                                                          kernel_size=(3, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7c/Branch_1/Conv2d_0b_1x3/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7c/Branch_1/Conv2d_0c_3x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0b_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7c/Branch_2/Conv2d_0b_3x1/BatchNorm/FusedBatchNorm', num_features=448,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0c_1x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7c/Branch_2/Conv2d_0c_1x3/Conv2D',
                                                                                          in_channels=448,
                                                                                          out_channels=512,
                                                                                          kernel_size=(1, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0c_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7c/Branch_2/Conv2d_0c_1x3/BatchNorm/FusedBatchNorm', num_features=512,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0d_1x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7c/Branch_2/Conv2d_0d_1x3/Conv2D',
                                                                                          in_channels=512,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0e_3x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7c/Branch_2/Conv2d_0e_3x1/Conv2D',
                                                                                          in_channels=512,
                                                                                          out_channels=256,
                                                                                          kernel_size=(3, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0d_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7c/Branch_2/Conv2d_0d_1x3/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0e_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7c/Branch_2/Conv2d_0e_3x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_0_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7d/Branch_0/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1536,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7d/Branch_1/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1536,
                                                                                          out_channels=384,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0a_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7d/Branch_2/Conv2d_0a_1x1/Conv2D',
                                                                                          in_channels=1536,
                                                                                          out_channels=384,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7d/Branch_0/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7d/Branch_1/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=384,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7d/Branch_2/Conv2d_0a_1x1/BatchNorm/FusedBatchNorm', num_features=384,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_3_Conv2d_0b_1x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7d/Branch_3/Conv2d_0b_1x1/Conv2D',
                                                                                          in_channels=1536,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7d/Branch_3/Conv2d_0b_1x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0b_1x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7d/Branch_1/Conv2d_0b_1x3/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0c_3x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7d/Branch_1/Conv2d_0c_3x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=256,
                                                                                          kernel_size=(3, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0b_3x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7d/Branch_2/Conv2d_0b_3x1/Conv2D',
                                                                                          in_channels=384,
                                                                                          out_channels=448,
                                                                                          kernel_size=(3, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7d/Branch_1/Conv2d_0b_1x3/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7d/Branch_1/Conv2d_0c_3x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0b_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7d/Branch_2/Conv2d_0b_3x1/BatchNorm/FusedBatchNorm', num_features=448,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0c_1x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7d/Branch_2/Conv2d_0c_1x3/Conv2D',
                                                                                          in_channels=448,
                                                                                          out_channels=512,
                                                                                          kernel_size=(1, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0c_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7d/Branch_2/Conv2d_0c_1x3/BatchNorm/FusedBatchNorm', num_features=512,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0d_1x3_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7d/Branch_2/Conv2d_0d_1x3/Conv2D',
                                                                                          in_channels=512,
                                                                                          out_channels=256,
                                                                                          kernel_size=(1, 3),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0e_3x1_Conv2D = self.__conv(2,
                                                                                          name='InceptionV4/InceptionV4/Mixed_7d/Branch_2/Conv2d_0e_3x1/Conv2D',
                                                                                          in_channels=512,
                                                                                          out_channels=256,
                                                                                          kernel_size=(3, 1),
                                                                                          stride=(1, 1), groups=1,
                                                                                          bias=None)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0d_1x3_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7d/Branch_2/Conv2d_0d_1x3/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0e_3x1_BatchNorm_FusedBatchNorm = self.__batch_normalization(
            2, 'InceptionV4/InceptionV4/Mixed_7d/Branch_2/Conv2d_0e_3x1/BatchNorm/FusedBatchNorm', num_features=256,
            eps=0.0010000000474974513, momentum=0.0)
        self.InceptionV4_Logits_Logits_MatMul = self.__dense(name='InceptionV4/Logits/Logits/MatMul', in_features=1536,
                                                             out_features=1001, bias=True)

    def forward(self, x):
        self.InceptionV4_AuxLogits_Flatten_flatten_Reshape_shape_1 = torch.autograd.Variable(torch.Tensor([-1]),
                                                                                             requires_grad=False)
        self.InceptionV4_Logits_PreLogitsFlatten_flatten_Reshape_shape_1 = torch.autograd.Variable(torch.Tensor([-1]),
                                                                                                   requires_grad=False)
        InceptionV4_InceptionV4_Conv2d_1a_3x3_Conv2D = self.InceptionV4_InceptionV4_Conv2d_1a_3x3_Conv2D(x)
        InceptionV4_InceptionV4_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Conv2d_1a_3x3_Conv2D)
        InceptionV4_InceptionV4_Conv2d_1a_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Conv2d_2a_3x3_Conv2D = self.InceptionV4_InceptionV4_Conv2d_2a_3x3_Conv2D(
            InceptionV4_InceptionV4_Conv2d_1a_3x3_Relu)
        InceptionV4_InceptionV4_Conv2d_2a_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Conv2d_2a_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Conv2d_2a_3x3_Conv2D)
        InceptionV4_InceptionV4_Conv2d_2a_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Conv2d_2a_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Conv2d_2b_3x3_Conv2D_pad = F.pad(InceptionV4_InceptionV4_Conv2d_2a_3x3_Relu,
                                                                 (1, 1, 1, 1))
        InceptionV4_InceptionV4_Conv2d_2b_3x3_Conv2D = self.InceptionV4_InceptionV4_Conv2d_2b_3x3_Conv2D(
            InceptionV4_InceptionV4_Conv2d_2b_3x3_Conv2D_pad)
        InceptionV4_InceptionV4_Conv2d_2b_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Conv2d_2b_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Conv2d_2b_3x3_Conv2D)
        InceptionV4_InceptionV4_Conv2d_2b_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Conv2d_2b_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_3a_Branch_0_MaxPool_0a_3x3_MaxPool, InceptionV4_InceptionV4_Mixed_3a_Branch_0_MaxPool_0a_3x3_MaxPool_idx = F.max_pool2d(
            InceptionV4_InceptionV4_Conv2d_2b_3x3_Relu, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False,
            return_indices=True)
        InceptionV4_InceptionV4_Mixed_3a_Branch_1_Conv2d_0a_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_3a_Branch_1_Conv2d_0a_3x3_Conv2D(
            InceptionV4_InceptionV4_Conv2d_2b_3x3_Relu)
        InceptionV4_InceptionV4_Mixed_3a_Branch_1_Conv2d_0a_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_3a_Branch_1_Conv2d_0a_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_3a_Branch_1_Conv2d_0a_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_3a_Branch_1_Conv2d_0a_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_3a_Branch_1_Conv2d_0a_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_3a_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_3a_Branch_0_MaxPool_0a_3x3_MaxPool,
                                                            InceptionV4_InceptionV4_Mixed_3a_Branch_1_Conv2d_0a_3x3_Relu,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_3a_concat)
        InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_3a_concat)
        InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_1a_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_1a_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_0a_1x1_Relu)
        InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_1a_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_1a_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_1a_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_1a_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_0c_7x1_Relu)
        InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_1a_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_1a_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_4a_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_4a_Branch_0_Conv2d_1a_3x3_Relu,
                                                            InceptionV4_InceptionV4_Mixed_4a_Branch_1_Conv2d_1a_3x3_Relu,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_5a_Branch_0_Conv2d_1a_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_5a_Branch_0_Conv2d_1a_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_4a_concat)
        InceptionV4_InceptionV4_Mixed_5a_Branch_1_MaxPool_1a_3x3_MaxPool, InceptionV4_InceptionV4_Mixed_5a_Branch_1_MaxPool_1a_3x3_MaxPool_idx = F.max_pool2d(
            InceptionV4_InceptionV4_Mixed_4a_concat, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False,
            return_indices=True)
        InceptionV4_InceptionV4_Mixed_5a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5a_Branch_0_Conv2d_1a_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_5a_Branch_0_Conv2d_1a_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5a_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_5a_Branch_0_Conv2d_1a_3x3_Relu,
                                                            InceptionV4_InceptionV4_Mixed_5a_Branch_1_MaxPool_1a_3x3_MaxPool,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_5b_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5b_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5a_concat)
        InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5a_concat)
        InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5a_concat)
        InceptionV4_InceptionV4_Mixed_5b_Branch_3_AvgPool_0a_3x3_AvgPool = F.avg_pool2d(
            InceptionV4_InceptionV4_Mixed_5a_concat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False,
            count_include_pad=False)
        InceptionV4_InceptionV4_Mixed_5b_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5b_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5b_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5b_Branch_3_AvgPool_0a_3x3_AvgPool)
        InceptionV4_InceptionV4_Mixed_5b_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5b_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5b_Branch_3_Conv2d_0b_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_5b_Branch_3_Conv2d_0b_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0c_3x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0b_3x3_Relu, (1, 1, 1, 1))
        InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0c_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0c_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0c_3x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0c_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0c_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5b_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_5b_Branch_0_Conv2d_0a_1x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_5b_Branch_1_Conv2d_0b_3x3_Relu,
                                                            InceptionV4_InceptionV4_Mixed_5b_Branch_2_Conv2d_0c_3x3_Relu,
                                                            InceptionV4_InceptionV4_Mixed_5b_Branch_3_Conv2d_0b_1x1_Relu,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_5c_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5c_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5b_concat)
        InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5b_concat)
        InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5b_concat)
        InceptionV4_InceptionV4_Mixed_5c_Branch_3_AvgPool_0a_3x3_AvgPool = F.avg_pool2d(
            InceptionV4_InceptionV4_Mixed_5b_concat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False,
            count_include_pad=False)
        InceptionV4_InceptionV4_Mixed_5c_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5c_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5c_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5c_Branch_3_Conv2d_0b_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5c_Branch_3_Conv2d_0b_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5c_Branch_3_AvgPool_0a_3x3_AvgPool)
        InceptionV4_InceptionV4_Mixed_5c_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5c_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5c_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5c_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5c_Branch_3_Conv2d_0b_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_5c_Branch_3_Conv2d_0b_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5c_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0c_3x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0b_3x3_Relu, (1, 1, 1, 1))
        InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0c_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0c_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0c_3x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0c_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0c_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5c_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_5c_Branch_0_Conv2d_0a_1x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_5c_Branch_1_Conv2d_0b_3x3_Relu,
                                                            InceptionV4_InceptionV4_Mixed_5c_Branch_2_Conv2d_0c_3x3_Relu,
                                                            InceptionV4_InceptionV4_Mixed_5c_Branch_3_Conv2d_0b_1x1_Relu,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_5d_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5d_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5c_concat)
        InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5c_concat)
        InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5c_concat)
        InceptionV4_InceptionV4_Mixed_5d_Branch_3_AvgPool_0a_3x3_AvgPool = F.avg_pool2d(
            InceptionV4_InceptionV4_Mixed_5c_concat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False,
            count_include_pad=False)
        InceptionV4_InceptionV4_Mixed_5d_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5d_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5d_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5d_Branch_3_Conv2d_0b_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5d_Branch_3_Conv2d_0b_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5d_Branch_3_AvgPool_0a_3x3_AvgPool)
        InceptionV4_InceptionV4_Mixed_5d_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5d_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5d_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5d_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5d_Branch_3_Conv2d_0b_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_5d_Branch_3_Conv2d_0b_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5d_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0c_3x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0b_3x3_Relu, (1, 1, 1, 1))
        InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0c_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0c_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0c_3x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0c_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0c_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5d_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_5d_Branch_0_Conv2d_0a_1x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_5d_Branch_1_Conv2d_0b_3x3_Relu,
                                                            InceptionV4_InceptionV4_Mixed_5d_Branch_2_Conv2d_0c_3x3_Relu,
                                                            InceptionV4_InceptionV4_Mixed_5d_Branch_3_Conv2d_0b_1x1_Relu,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_5e_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5e_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5d_concat)
        InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5d_concat)
        InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5d_concat)
        InceptionV4_InceptionV4_Mixed_5e_Branch_3_AvgPool_0a_3x3_AvgPool = F.avg_pool2d(
            InceptionV4_InceptionV4_Mixed_5d_concat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False,
            count_include_pad=False)
        InceptionV4_InceptionV4_Mixed_5e_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5e_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5e_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5e_Branch_3_Conv2d_0b_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_5e_Branch_3_Conv2d_0b_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5e_Branch_3_AvgPool_0a_3x3_AvgPool)
        InceptionV4_InceptionV4_Mixed_5e_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5e_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5e_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5e_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5e_Branch_3_Conv2d_0b_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0b_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0b_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0b_3x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_5e_Branch_3_Conv2d_0b_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5e_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0b_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0b_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0c_3x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0b_3x3_Relu, (1, 1, 1, 1))
        InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0c_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0c_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0c_3x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0c_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0c_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0c_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_5e_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_5e_Branch_0_Conv2d_0a_1x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_5e_Branch_1_Conv2d_0b_3x3_Relu,
                                                            InceptionV4_InceptionV4_Mixed_5e_Branch_2_Conv2d_0c_3x3_Relu,
                                                            InceptionV4_InceptionV4_Mixed_5e_Branch_3_Conv2d_0b_1x1_Relu,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_6a_Branch_0_Conv2d_1a_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_6a_Branch_0_Conv2d_1a_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_5e_concat)
        InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_5e_concat)
        InceptionV4_InceptionV4_Mixed_6a_Branch_2_MaxPool_1a_3x3_MaxPool, InceptionV4_InceptionV4_Mixed_6a_Branch_2_MaxPool_1a_3x3_MaxPool_idx = F.max_pool2d(
            InceptionV4_InceptionV4_Mixed_5e_concat, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False,
            return_indices=True)
        InceptionV4_InceptionV4_Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6a_Branch_0_Conv2d_1a_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6a_Branch_0_Conv2d_1a_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0b_3x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 1, 1))
        InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0b_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0b_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0b_3x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0b_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0b_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_1a_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_1a_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_0b_3x3_Relu)
        InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_1a_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_1a_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6a_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_6a_Branch_0_Conv2d_1a_3x3_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6a_Branch_1_Conv2d_1a_3x3_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6a_Branch_2_MaxPool_1a_3x3_MaxPool,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_6b_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6b_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6a_concat)
        InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6a_concat)
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6a_concat)
        InceptionV4_InceptionV4_Mixed_6b_Branch_3_AvgPool_0a_3x3_AvgPool = F.avg_pool2d(
            InceptionV4_InceptionV4_Mixed_6a_concat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False,
            count_include_pad=False)
        InceptionV4_InceptionV4_Mixed_6b_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6b_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6b_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6b_Branch_3_Conv2d_0b_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6b_Branch_3_Conv2d_0b_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6b_Branch_3_AvgPool_0a_3x3_AvgPool)
        InceptionV4_InceptionV4_Mixed_6b_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6b_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6b_Branch_3_Conv2d_0b_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0b_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0a_1x1_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0b_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0b_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0b_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6b_Branch_3_Conv2d_0b_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0b_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0b_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0c_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0b_7x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0c_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0c_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0c_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0c_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0c_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0d_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0c_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0d_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0d_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0d_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0d_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0d_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0e_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0d_7x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0e_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0e_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0e_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0e_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0e_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6b_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_6b_Branch_0_Conv2d_0a_1x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6b_Branch_1_Conv2d_0c_7x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6b_Branch_2_Conv2d_0e_1x7_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6b_Branch_3_Conv2d_0b_1x1_Relu,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_6c_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6c_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6b_concat)
        InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6b_concat)
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6b_concat)
        InceptionV4_InceptionV4_Mixed_6c_Branch_3_AvgPool_0a_3x3_AvgPool = F.avg_pool2d(
            InceptionV4_InceptionV4_Mixed_6b_concat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False,
            count_include_pad=False)
        InceptionV4_InceptionV4_Mixed_6c_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6c_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6c_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6c_Branch_3_Conv2d_0b_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6c_Branch_3_Conv2d_0b_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6c_Branch_3_AvgPool_0a_3x3_AvgPool)
        InceptionV4_InceptionV4_Mixed_6c_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6c_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6c_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6c_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6c_Branch_3_Conv2d_0b_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0b_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0a_1x1_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0b_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0b_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0b_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6c_Branch_3_Conv2d_0b_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6c_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0b_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0b_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0c_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0b_7x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0c_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0c_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0c_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0c_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0c_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0d_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0c_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0d_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0d_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0d_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0d_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0d_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0e_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0d_7x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0e_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0e_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0e_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0e_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0e_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6c_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_6c_Branch_0_Conv2d_0a_1x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6c_Branch_1_Conv2d_0c_7x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6c_Branch_2_Conv2d_0e_1x7_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6c_Branch_3_Conv2d_0b_1x1_Relu,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_6d_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6d_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6c_concat)
        InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6c_concat)
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6c_concat)
        InceptionV4_InceptionV4_Mixed_6d_Branch_3_AvgPool_0a_3x3_AvgPool = F.avg_pool2d(
            InceptionV4_InceptionV4_Mixed_6c_concat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False,
            count_include_pad=False)
        InceptionV4_InceptionV4_Mixed_6d_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6d_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6d_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6d_Branch_3_Conv2d_0b_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6d_Branch_3_Conv2d_0b_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6d_Branch_3_AvgPool_0a_3x3_AvgPool)
        InceptionV4_InceptionV4_Mixed_6d_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6d_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6d_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6d_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6d_Branch_3_Conv2d_0b_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0b_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0a_1x1_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0b_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0b_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0b_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6d_Branch_3_Conv2d_0b_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6d_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0b_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0b_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0c_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0b_7x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0c_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0c_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0c_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0c_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0c_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0d_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0c_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0d_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0d_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0d_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0d_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0d_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0e_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0d_7x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0e_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0e_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0e_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0e_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0e_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6d_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_6d_Branch_0_Conv2d_0a_1x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6d_Branch_1_Conv2d_0c_7x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6d_Branch_2_Conv2d_0e_1x7_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6d_Branch_3_Conv2d_0b_1x1_Relu,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_6e_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6e_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6d_concat)
        InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6d_concat)
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6d_concat)
        InceptionV4_InceptionV4_Mixed_6e_Branch_3_AvgPool_0a_3x3_AvgPool = F.avg_pool2d(
            InceptionV4_InceptionV4_Mixed_6d_concat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False,
            count_include_pad=False)
        InceptionV4_InceptionV4_Mixed_6e_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6e_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6e_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6e_Branch_3_Conv2d_0b_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6e_Branch_3_Conv2d_0b_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6e_Branch_3_AvgPool_0a_3x3_AvgPool)
        InceptionV4_InceptionV4_Mixed_6e_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6e_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6e_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6e_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6e_Branch_3_Conv2d_0b_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0b_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0a_1x1_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0b_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0b_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0b_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6e_Branch_3_Conv2d_0b_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6e_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0b_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0b_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0c_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0b_7x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0c_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0c_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0c_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0c_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0c_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0d_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0c_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0d_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0d_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0d_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0d_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0d_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0e_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0d_7x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0e_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0e_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0e_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0e_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0e_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6e_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_6e_Branch_0_Conv2d_0a_1x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6e_Branch_1_Conv2d_0c_7x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6e_Branch_2_Conv2d_0e_1x7_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6e_Branch_3_Conv2d_0b_1x1_Relu,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_6f_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6f_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6e_concat)
        InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6e_concat)
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6e_concat)
        InceptionV4_InceptionV4_Mixed_6f_Branch_3_AvgPool_0a_3x3_AvgPool = F.avg_pool2d(
            InceptionV4_InceptionV4_Mixed_6e_concat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False,
            count_include_pad=False)
        InceptionV4_InceptionV4_Mixed_6f_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6f_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6f_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6f_Branch_3_Conv2d_0b_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6f_Branch_3_Conv2d_0b_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6f_Branch_3_AvgPool_0a_3x3_AvgPool)
        InceptionV4_InceptionV4_Mixed_6f_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6f_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6f_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6f_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6f_Branch_3_Conv2d_0b_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0b_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0a_1x1_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0b_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0b_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0b_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6f_Branch_3_Conv2d_0b_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6f_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0b_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0b_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0c_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0b_7x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0c_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0c_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0c_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0c_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0c_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0d_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0c_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0d_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0d_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0d_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0d_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0d_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0e_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0d_7x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0e_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0e_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0e_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0e_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0e_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6f_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_6f_Branch_0_Conv2d_0a_1x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6f_Branch_1_Conv2d_0c_7x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6f_Branch_2_Conv2d_0e_1x7_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6f_Branch_3_Conv2d_0b_1x1_Relu,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_6g_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6g_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6f_concat)
        InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6f_concat)
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6f_concat)
        InceptionV4_InceptionV4_Mixed_6g_Branch_3_AvgPool_0a_3x3_AvgPool = F.avg_pool2d(
            InceptionV4_InceptionV4_Mixed_6f_concat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False,
            count_include_pad=False)
        InceptionV4_InceptionV4_Mixed_6g_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6g_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6g_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6g_Branch_3_Conv2d_0b_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6g_Branch_3_Conv2d_0b_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6g_Branch_3_AvgPool_0a_3x3_AvgPool)
        InceptionV4_InceptionV4_Mixed_6g_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6g_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6g_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6g_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6g_Branch_3_Conv2d_0b_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0b_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0a_1x1_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0b_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0b_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0b_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6g_Branch_3_Conv2d_0b_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6g_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0b_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0b_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0c_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0b_7x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0c_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0c_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0c_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0c_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0c_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0d_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0c_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0d_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0d_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0d_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0d_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0d_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0e_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0d_7x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0e_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0e_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0e_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0e_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0e_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6g_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_6g_Branch_0_Conv2d_0a_1x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6g_Branch_1_Conv2d_0c_7x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6g_Branch_2_Conv2d_0e_1x7_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6g_Branch_3_Conv2d_0b_1x1_Relu,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_6h_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6h_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6g_concat)
        InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6g_concat)
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6g_concat)
        InceptionV4_InceptionV4_Mixed_6h_Branch_3_AvgPool_0a_3x3_AvgPool = F.avg_pool2d(
            InceptionV4_InceptionV4_Mixed_6g_concat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False,
            count_include_pad=False)
        InceptionV4_InceptionV4_Mixed_6h_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6h_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6h_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6h_Branch_3_Conv2d_0b_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6h_Branch_3_Conv2d_0b_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6h_Branch_3_AvgPool_0a_3x3_AvgPool)
        InceptionV4_InceptionV4_Mixed_6h_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6h_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6h_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6h_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6h_Branch_3_Conv2d_0b_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0b_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0a_1x1_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0b_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0b_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0b_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6h_Branch_3_Conv2d_0b_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6h_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0b_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0b_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0b_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0c_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0b_7x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0c_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0c_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0c_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0c_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0c_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0c_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0d_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0c_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0d_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0d_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0d_7x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0d_7x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0d_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0d_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0e_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0d_7x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0e_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0e_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0e_1x7_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0e_1x7_Conv2D)
        InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0e_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0e_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_6h_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_6h_Branch_0_Conv2d_0a_1x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6h_Branch_1_Conv2d_0c_7x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6h_Branch_2_Conv2d_0e_1x7_Relu,
                                                            InceptionV4_InceptionV4_Mixed_6h_Branch_3_Conv2d_0b_1x1_Relu,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6h_concat)
        InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_6h_concat)
        InceptionV4_InceptionV4_Mixed_7a_Branch_2_MaxPool_1a_3x3_MaxPool, InceptionV4_InceptionV4_Mixed_7a_Branch_2_MaxPool_1a_3x3_MaxPool_idx = F.max_pool2d(
            InceptionV4_InceptionV4_Mixed_6h_concat, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False,
            return_indices=True)
        InceptionV4_AuxLogits_AvgPool_1a_5x5_AvgPool = F.avg_pool2d(InceptionV4_InceptionV4_Mixed_6h_concat,
                                                                    kernel_size=(5, 5), stride=(3, 3), padding=(0,),
                                                                    ceil_mode=False, count_include_pad=False)
        InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_AuxLogits_Conv2d_1b_1x1_Conv2D = self.InceptionV4_AuxLogits_Conv2d_1b_1x1_Conv2D(
            InceptionV4_AuxLogits_AvgPool_1a_5x5_AvgPool)
        InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_AuxLogits_Conv2d_1b_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_AuxLogits_Conv2d_1b_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_AuxLogits_Conv2d_1b_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_1a_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_1a_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_0a_1x1_Relu)
        InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0b_1x7_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0a_1x1_Relu, (3, 3, 0, 0))
        InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0b_1x7_Conv2D = self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0b_1x7_Conv2D(
            InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0b_1x7_Conv2D_pad)
        InceptionV4_AuxLogits_Conv2d_1b_1x1_Relu = F.relu(InceptionV4_AuxLogits_Conv2d_1b_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_1a_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0b_1x7_Conv2D)
        InceptionV4_AuxLogits_Conv2d_2a_Conv2D = self.InceptionV4_AuxLogits_Conv2d_2a_Conv2D(
            InceptionV4_AuxLogits_Conv2d_1b_1x1_Relu)
        InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_1a_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0b_1x7_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0b_1x7_BatchNorm_FusedBatchNorm)
        InceptionV4_AuxLogits_Conv2d_2a_BatchNorm_FusedBatchNorm = self.InceptionV4_AuxLogits_Conv2d_2a_BatchNorm_FusedBatchNorm(
            InceptionV4_AuxLogits_Conv2d_2a_Conv2D)
        InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0c_7x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0b_1x7_Relu, (0, 0, 3, 3))
        InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0c_7x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0c_7x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0c_7x1_Conv2D_pad)
        InceptionV4_AuxLogits_Conv2d_2a_Relu = F.relu(InceptionV4_AuxLogits_Conv2d_2a_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0c_7x1_Conv2D)
        InceptionV4_AuxLogits_Flatten_flatten_Shape = torch.Tensor(list(InceptionV4_AuxLogits_Conv2d_2a_Relu.size()))
        InceptionV4_AuxLogits_Flatten_flatten_Reshape = torch.reshape(input=InceptionV4_AuxLogits_Conv2d_2a_Relu,
                                                                      shape=(-1, 768))
        InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0c_7x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0c_7x1_BatchNorm_FusedBatchNorm)
        InceptionV4_AuxLogits_Flatten_flatten_strided_slice = InceptionV4_AuxLogits_Flatten_flatten_Shape[0:1][0]
        InceptionV4_AuxLogits_Aux_logits_MatMul = self.InceptionV4_AuxLogits_Aux_logits_MatMul(
            InceptionV4_AuxLogits_Flatten_flatten_Reshape)
        InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_1a_3x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_1a_3x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_0c_7x1_Relu)
        InceptionV4_AuxLogits_Flatten_flatten_Reshape_shape = [InceptionV4_AuxLogits_Flatten_flatten_strided_slice,
                                                               self.InceptionV4_AuxLogits_Flatten_flatten_Reshape_shape_1]
        InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_1a_3x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_1a_3x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7a_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_7a_Branch_0_Conv2d_1a_3x3_Relu,
                                                            InceptionV4_InceptionV4_Mixed_7a_Branch_1_Conv2d_1a_3x3_Relu,
                                                            InceptionV4_InceptionV4_Mixed_7a_Branch_2_MaxPool_1a_3x3_MaxPool,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_7b_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7b_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7a_concat)
        InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7a_concat)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7a_concat)
        InceptionV4_InceptionV4_Mixed_7b_Branch_3_AvgPool_0a_3x3_AvgPool = F.avg_pool2d(
            InceptionV4_InceptionV4_Mixed_7a_concat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False,
            count_include_pad=False)
        InceptionV4_InceptionV4_Mixed_7b_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7b_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7b_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7b_Branch_3_Conv2d_0b_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7b_Branch_3_Conv2d_0b_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7b_Branch_3_AvgPool_0a_3x3_AvgPool)
        InceptionV4_InceptionV4_Mixed_7b_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7b_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7b_Branch_3_Conv2d_0b_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0b_1x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 0, 0))
        InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0b_1x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0b_1x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0b_1x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0c_3x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0a_1x1_Relu, (0, 0, 1, 1))
        InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0c_3x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0c_3x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0c_3x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0b_3x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0a_1x1_Relu, (0, 0, 1, 1))
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0b_3x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0b_3x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0b_3x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7b_Branch_3_Conv2d_0b_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7b_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0b_1x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0c_3x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0b_3x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0b_3x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0b_3x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0b_1x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0c_3x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0b_3x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0b_3x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7b_Branch_1_concat = torch.cat((
                                                                     InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0b_1x3_Relu,
                                                                     InceptionV4_InceptionV4_Mixed_7b_Branch_1_Conv2d_0c_3x1_Relu,),
                                                                     1)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0c_1x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0b_3x1_Relu, (1, 1, 0, 0))
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0c_1x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0c_1x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0c_1x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0c_1x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0c_1x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0c_1x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0c_1x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0c_1x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0d_1x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0c_1x3_Relu, (1, 1, 0, 0))
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0d_1x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0d_1x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0d_1x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0e_3x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0c_1x3_Relu, (0, 0, 1, 1))
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0e_3x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0e_3x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0e_3x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0d_1x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0d_1x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0d_1x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0e_3x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0e_3x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0e_3x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0d_1x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0d_1x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0e_3x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0e_3x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7b_Branch_2_concat = torch.cat((
                                                                     InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0d_1x3_Relu,
                                                                     InceptionV4_InceptionV4_Mixed_7b_Branch_2_Conv2d_0e_3x1_Relu,),
                                                                     1)
        InceptionV4_InceptionV4_Mixed_7b_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_7b_Branch_0_Conv2d_0a_1x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_7b_Branch_1_concat,
                                                            InceptionV4_InceptionV4_Mixed_7b_Branch_2_concat,
                                                            InceptionV4_InceptionV4_Mixed_7b_Branch_3_Conv2d_0b_1x1_Relu,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_7c_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7c_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7b_concat)
        InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7b_concat)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7b_concat)
        InceptionV4_InceptionV4_Mixed_7c_Branch_3_AvgPool_0a_3x3_AvgPool = F.avg_pool2d(
            InceptionV4_InceptionV4_Mixed_7b_concat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False,
            count_include_pad=False)
        InceptionV4_InceptionV4_Mixed_7c_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7c_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7c_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7c_Branch_3_Conv2d_0b_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7c_Branch_3_Conv2d_0b_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7c_Branch_3_AvgPool_0a_3x3_AvgPool)
        InceptionV4_InceptionV4_Mixed_7c_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7c_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7c_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7c_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7c_Branch_3_Conv2d_0b_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0b_1x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 0, 0))
        InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0b_1x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0b_1x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0b_1x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0c_3x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0a_1x1_Relu, (0, 0, 1, 1))
        InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0c_3x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0c_3x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0c_3x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0b_3x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0a_1x1_Relu, (0, 0, 1, 1))
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0b_3x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0b_3x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0b_3x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7c_Branch_3_Conv2d_0b_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7c_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0b_1x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0c_3x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0b_3x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0b_3x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0b_3x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0b_1x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0c_3x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0b_3x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0b_3x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7c_Branch_1_concat = torch.cat((
                                                                     InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0b_1x3_Relu,
                                                                     InceptionV4_InceptionV4_Mixed_7c_Branch_1_Conv2d_0c_3x1_Relu,),
                                                                     1)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0c_1x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0b_3x1_Relu, (1, 1, 0, 0))
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0c_1x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0c_1x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0c_1x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0c_1x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0c_1x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0c_1x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0c_1x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0c_1x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0d_1x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0c_1x3_Relu, (1, 1, 0, 0))
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0d_1x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0d_1x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0d_1x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0e_3x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0c_1x3_Relu, (0, 0, 1, 1))
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0e_3x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0e_3x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0e_3x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0d_1x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0d_1x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0d_1x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0e_3x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0e_3x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0e_3x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0d_1x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0d_1x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0e_3x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0e_3x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7c_Branch_2_concat = torch.cat((
                                                                     InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0d_1x3_Relu,
                                                                     InceptionV4_InceptionV4_Mixed_7c_Branch_2_Conv2d_0e_3x1_Relu,),
                                                                     1)
        InceptionV4_InceptionV4_Mixed_7c_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_7c_Branch_0_Conv2d_0a_1x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_7c_Branch_1_concat,
                                                            InceptionV4_InceptionV4_Mixed_7c_Branch_2_concat,
                                                            InceptionV4_InceptionV4_Mixed_7c_Branch_3_Conv2d_0b_1x1_Relu,),
                                                            1)
        InceptionV4_InceptionV4_Mixed_7d_Branch_0_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7d_Branch_0_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7c_concat)
        InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7c_concat)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0a_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0a_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7c_concat)
        InceptionV4_InceptionV4_Mixed_7d_Branch_3_AvgPool_0a_3x3_AvgPool = F.avg_pool2d(
            InceptionV4_InceptionV4_Mixed_7c_concat, kernel_size=(3, 3), stride=(1, 1), padding=(1,), ceil_mode=False,
            count_include_pad=False)
        InceptionV4_InceptionV4_Mixed_7d_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7d_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7d_Branch_0_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0a_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7d_Branch_3_Conv2d_0b_1x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7d_Branch_3_Conv2d_0b_1x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7d_Branch_3_AvgPool_0a_3x3_AvgPool)
        InceptionV4_InceptionV4_Mixed_7d_Branch_0_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7d_Branch_0_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0a_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0a_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7d_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7d_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7d_Branch_3_Conv2d_0b_1x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0b_1x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0a_1x1_Relu, (1, 1, 0, 0))
        InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0b_1x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0b_1x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0b_1x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0c_3x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0a_1x1_Relu, (0, 0, 1, 1))
        InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0c_3x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0c_3x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0c_3x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0b_3x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0a_1x1_Relu, (0, 0, 1, 1))
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0b_3x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0b_3x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0b_3x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7d_Branch_3_Conv2d_0b_1x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7d_Branch_3_Conv2d_0b_1x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0b_1x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0c_3x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0b_3x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0b_3x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0b_3x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0b_1x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0b_1x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0c_3x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0c_3x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0b_3x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0b_3x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7d_Branch_1_concat = torch.cat((
                                                                     InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0b_1x3_Relu,
                                                                     InceptionV4_InceptionV4_Mixed_7d_Branch_1_Conv2d_0c_3x1_Relu,),
                                                                     1)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0c_1x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0b_3x1_Relu, (1, 1, 0, 0))
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0c_1x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0c_1x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0c_1x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0c_1x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0c_1x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0c_1x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0c_1x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0c_1x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0d_1x3_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0c_1x3_Relu, (1, 1, 0, 0))
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0d_1x3_Conv2D = self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0d_1x3_Conv2D(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0d_1x3_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0e_3x1_Conv2D_pad = F.pad(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0c_1x3_Relu, (0, 0, 1, 1))
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0e_3x1_Conv2D = self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0e_3x1_Conv2D(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0e_3x1_Conv2D_pad)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0d_1x3_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0d_1x3_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0d_1x3_Conv2D)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0e_3x1_BatchNorm_FusedBatchNorm = self.InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0e_3x1_BatchNorm_FusedBatchNorm(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0e_3x1_Conv2D)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0d_1x3_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0d_1x3_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0e_3x1_Relu = F.relu(
            InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0e_3x1_BatchNorm_FusedBatchNorm)
        InceptionV4_InceptionV4_Mixed_7d_Branch_2_concat = torch.cat((
                                                                     InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0d_1x3_Relu,
                                                                     InceptionV4_InceptionV4_Mixed_7d_Branch_2_Conv2d_0e_3x1_Relu,),
                                                                     1)
        InceptionV4_InceptionV4_Mixed_7d_concat = torch.cat((
                                                            InceptionV4_InceptionV4_Mixed_7d_Branch_0_Conv2d_0a_1x1_Relu,
                                                            InceptionV4_InceptionV4_Mixed_7d_Branch_1_concat,
                                                            InceptionV4_InceptionV4_Mixed_7d_Branch_2_concat,
                                                            InceptionV4_InceptionV4_Mixed_7d_Branch_3_Conv2d_0b_1x1_Relu,),
                                                            1)
        kernel_size = self._reduced_kernel_size_for_small_input(InceptionV4_InceptionV4_Mixed_7d_concat, [8, 8])
        InceptionV4_Logits_AvgPool_1a_AvgPool = F.avg_pool2d(InceptionV4_InceptionV4_Mixed_7d_concat,
                                                             kernel_size=(kernel_size[0], kernel_size[1]),
                                                             stride=(1, 1), padding=(0,), ceil_mode=False,
                                                             count_include_pad=False)
        InceptionV4_Logits_PreLogitsFlatten_flatten_Shape = torch.Tensor(
            list(InceptionV4_Logits_AvgPool_1a_AvgPool.size()))
        InceptionV4_Logits_PreLogitsFlatten_flatten_Reshape = torch.reshape(input=InceptionV4_Logits_AvgPool_1a_AvgPool,
                                                                            shape=(-1, 1536))
        InceptionV4_Logits_PreLogitsFlatten_flatten_strided_slice = \
        InceptionV4_Logits_PreLogitsFlatten_flatten_Shape[0:1][0]
        InceptionV4_Logits_Logits_MatMul = self.InceptionV4_Logits_Logits_MatMul(
            InceptionV4_Logits_PreLogitsFlatten_flatten_Reshape)
        InceptionV4_Logits_PreLogitsFlatten_flatten_Reshape_shape = [
            InceptionV4_Logits_PreLogitsFlatten_flatten_strided_slice,
            self.InceptionV4_Logits_PreLogitsFlatten_flatten_Reshape_shape_1]
        MMdnn_Output_input = [InceptionV4_Logits_Logits_MatMul, InceptionV4_AuxLogits_Aux_logits_MatMul]
        result = InceptionV4_Logits_Logits_MatMul[:, 1:]
        return result

    def _reduced_kernel_size_for_small_input(self, input_tensor, kernel_size):
        """Define kernel size which is automatically reduced for small input.

        If the shape of the input images is unknown at graph construction time this
        function assumes that the input images are is large enough.

        Args:
            input_tensor: input tensor of size [batch_size, height, width, channels].
            kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

        Returns:
            a tensor with the kernel size.

        """
        shape = input_tensor.shape
        if shape[2] is None or shape[3] is None:
            kernel_size_out = kernel_size
        else:
            kernel_size_out = [min(shape[2], kernel_size[0]),
                               min(shape[3], kernel_size[1])]
        return kernel_size_out

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if dim == 1:
            layer = nn.Conv1d(**kwargs)
        elif dim == 2:
            layer = nn.Conv2d(**kwargs)
        elif dim == 3:
            layer = nn.Conv3d(**kwargs)
        else:
            raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if dim == 0 or dim == 1:
            layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:
            layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:
            layer = nn.BatchNorm3d(**kwargs)
        else:
            raise NotImplementedError()

        if 'scale' in _weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(_weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(_weights_dict[name]['var']))
        return layer
