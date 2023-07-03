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


class Res152KitModel(nn.Module):

    def __init__(self, weight_file):
        super(Res152KitModel, self).__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)

        self.resnet_v2_152_conv1_Conv2D = self.__conv(2, name='resnet_v2_152/conv1/Conv2D', in_channels=3,
                                                      out_channels=64, kernel_size=(7, 7), stride=(2, 2), groups=1,
                                                      bias=True)
        self.resnet_v2_152_block1_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block1/unit_1/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=64,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block1_unit_1_bottleneck_v2_shortcut_Conv2D = self.__conv(2,
                                                                                     name='resnet_v2_152/block1/unit_1/bottleneck_v2/shortcut/Conv2D',
                                                                                     in_channels=64, out_channels=256,
                                                                                     kernel_size=(1, 1), stride=(1, 1),
                                                                                     groups=1, bias=True)
        self.resnet_v2_152_block1_unit_1_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block1/unit_1/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=64, out_channels=64,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block1_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block1/unit_1/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=64,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block1_unit_1_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block1/unit_1/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=64, out_channels=64,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block1_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block1/unit_1/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=64,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block1_unit_1_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block1/unit_1/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=64, out_channels=256,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block1_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block1/unit_2/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=256,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block1_unit_2_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block1/unit_2/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=256, out_channels=64,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block1_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block1/unit_2/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=64,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block1_unit_2_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block1/unit_2/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=64, out_channels=64,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block1_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block1/unit_2/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=64,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block1_unit_2_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block1/unit_2/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=64, out_channels=256,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block1_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block1/unit_3/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=256,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block1_unit_3_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block1/unit_3/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=256, out_channels=64,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block1_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block1/unit_3/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=64,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block1_unit_3_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block1/unit_3/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=64, out_channels=64,
                                                                                  kernel_size=(3, 3), stride=(2, 2),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block1_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block1/unit_3/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=64,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block1_unit_3_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block1/unit_3/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=64, out_channels=256,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block2_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block2/unit_1/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=256,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block2_unit_1_bottleneck_v2_shortcut_Conv2D = self.__conv(2,
                                                                                     name='resnet_v2_152/block2/unit_1/bottleneck_v2/shortcut/Conv2D',
                                                                                     in_channels=256, out_channels=512,
                                                                                     kernel_size=(1, 1), stride=(1, 1),
                                                                                     groups=1, bias=True)
        self.resnet_v2_152_block2_unit_1_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_1/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=256, out_channels=128,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_1/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_1_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_1/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=128, out_channels=128,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_1/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_1_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_1/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=128, out_channels=512,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block2_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block2/unit_2/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=512,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block2_unit_2_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_2/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=512, out_channels=128,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_2/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_2_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_2/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=128, out_channels=128,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_2/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_2_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_2/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=128, out_channels=512,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block2_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block2/unit_3/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=512,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block2_unit_3_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_3/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=512, out_channels=128,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_3/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_3_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_3/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=128, out_channels=128,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_3/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_3_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_3/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=128, out_channels=512,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block2_unit_4_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block2/unit_4/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=512,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block2_unit_4_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_4/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=512, out_channels=128,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_4/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_4_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_4/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=128, out_channels=128,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_4/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_4_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_4/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=128, out_channels=512,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block2_unit_5_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block2/unit_5/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=512,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block2_unit_5_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_5/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=512, out_channels=128,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_5/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_5_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_5/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=128, out_channels=128,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_5/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_5_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_5/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=128, out_channels=512,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block2_unit_6_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block2/unit_6/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=512,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block2_unit_6_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_6/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=512, out_channels=128,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_6/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_6_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_6/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=128, out_channels=128,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_6/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_6_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_6/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=128, out_channels=512,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block2_unit_7_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block2/unit_7/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=512,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block2_unit_7_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_7/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=512, out_channels=128,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_7_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_7/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_7_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_7/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=128, out_channels=128,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_7_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_7/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_7_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_7/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=128, out_channels=512,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block2_unit_8_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block2/unit_8/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=512,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block2_unit_8_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_8/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=512, out_channels=128,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_8_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_8/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_8_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_8/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=128, out_channels=128,
                                                                                  kernel_size=(3, 3), stride=(2, 2),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block2_unit_8_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block2/unit_8/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=128,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block2_unit_8_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block2/unit_8/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=128, out_channels=512,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block3_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block3/unit_1/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=512,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block3_unit_1_bottleneck_v2_shortcut_Conv2D = self.__conv(2,
                                                                                     name='resnet_v2_152/block3/unit_1/bottleneck_v2/shortcut/Conv2D',
                                                                                     in_channels=512, out_channels=1024,
                                                                                     kernel_size=(1, 1), stride=(1, 1),
                                                                                     groups=1, bias=True)
        self.resnet_v2_152_block3_unit_1_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_1/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=512, out_channels=256,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_1/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_1_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_1/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=256, out_channels=256,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_1/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_1_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_1/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=256, out_channels=1024,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block3_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block3/unit_2/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=1024,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block3_unit_2_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_2/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=1024, out_channels=256,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_2/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_2_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_2/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=256, out_channels=256,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_2/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_2_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_2/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=256, out_channels=1024,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block3_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block3/unit_3/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=1024,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block3_unit_3_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_3/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=1024, out_channels=256,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_3/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_3_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_3/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=256, out_channels=256,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_3/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_3_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_3/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=256, out_channels=1024,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block3_unit_4_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block3/unit_4/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=1024,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block3_unit_4_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_4/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=1024, out_channels=256,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_4/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_4_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_4/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=256, out_channels=256,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_4/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_4_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_4/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=256, out_channels=1024,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block3_unit_5_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block3/unit_5/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=1024,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block3_unit_5_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_5/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=1024, out_channels=256,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_5/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_5_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_5/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=256, out_channels=256,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_5/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_5_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_5/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=256, out_channels=1024,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block3_unit_6_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block3/unit_6/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=1024,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block3_unit_6_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_6/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=1024, out_channels=256,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_6/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_6_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_6/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=256, out_channels=256,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_6/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_6_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_6/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=256, out_channels=1024,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block3_unit_7_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block3/unit_7/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=1024,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block3_unit_7_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_7/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=1024, out_channels=256,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_7_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_7/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_7_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_7/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=256, out_channels=256,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_7_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_7/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_7_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_7/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=256, out_channels=1024,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block3_unit_8_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block3/unit_8/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=1024,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block3_unit_8_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_8/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=1024, out_channels=256,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_8_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_8/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_8_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_8/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=256, out_channels=256,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_8_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_8/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_8_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_8/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=256, out_channels=1024,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block3_unit_9_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block3/unit_9/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=1024,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block3_unit_9_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_9/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=1024, out_channels=256,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_9_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_9/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_9_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_9/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=256, out_channels=256,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block3_unit_9_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block3/unit_9/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=256,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block3_unit_9_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block3/unit_9/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=256, out_channels=1024,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block3_unit_10_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_10/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_10_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_10/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_10_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_10/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_10_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_10/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_10_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_10/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_10_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_10/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_11_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_11/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_11_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_11/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_11_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_11/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_11_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_11/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_11_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_11/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_11_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_11/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_12_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_12/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_12_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_12/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_12_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_12/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_12_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_12/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_12_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_12/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_12_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_12/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_13_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_13/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_13_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_13/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_13_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_13/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_13_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_13/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_13_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_13/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_13_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_13/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_14_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_14/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_14_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_14/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_14_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_14/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_14_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_14/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_14_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_14/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_14_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_14/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_15_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_15/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_15_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_15/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_15_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_15/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_15_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_15/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_15_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_15/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_15_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_15/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_16_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_16/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_16_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_16/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_16_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_16/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_16_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_16/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_16_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_16/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_16_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_16/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_17_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_17/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_17_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_17/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_17_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_17/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_17_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_17/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_17_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_17/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_17_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_17/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_18_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_18/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_18_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_18/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_18_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_18/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_18_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_18/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_18_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_18/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_18_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_18/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_19_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_19/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_19_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_19/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_19_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_19/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_19_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_19/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_19_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_19/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_19_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_19/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_20_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_20/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_20_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_20/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_20_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_20/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_20_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_20/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_20_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_20/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_20_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_20/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_21_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_21/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_21_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_21/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_21_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_21/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_21_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_21/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_21_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_21/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_21_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_21/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_22_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_22/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_22_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_22/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_22_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_22/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_22_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_22/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_22_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_22/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_22_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_22/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_23_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_23/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_23_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_23/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_23_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_23/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_23_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_23/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_23_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_23/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_23_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_23/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_24_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_24/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_24_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_24/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_24_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_24/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_24_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_24/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_24_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_24/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_24_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_24/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_25_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_25/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_25_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_25/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_25_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_25/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_25_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_25/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_25_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_25/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_25_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_25/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_26_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_26/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_26_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_26/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_26_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_26/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_26_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_26/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_26_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_26/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_26_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_26/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_27_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_27/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_27_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_27/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_27_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_27/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_27_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_27/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_27_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_27/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_27_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_27/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_28_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_28/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_28_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_28/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_28_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_28/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_28_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_28/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_28_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_28/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_28_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_28/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_29_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_29/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_29_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_29/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_29_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_29/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_29_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_29/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_29_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_29/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_29_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_29/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_30_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_30/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_30_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_30/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_30_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_30/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_30_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_30/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_30_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_30/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_30_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_30/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_31_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_31/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_31_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_31/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_31_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_31/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_31_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_31/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_31_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_31/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_31_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_31/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_32_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_32/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_32_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_32/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_32_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_32/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_32_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_32/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_32_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_32/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_32_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_32/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_33_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_33/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_33_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_33/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_33_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_33/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_33_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_33/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_33_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_33/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_33_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_33/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_34_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_34/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_34_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_34/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_34_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_34/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_34_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_34/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_34_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_34/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_34_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_34/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_35_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_35/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_35_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_35/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_35_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_35/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_35_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_35/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_35_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_35/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_35_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_35/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block3_unit_36_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                           'resnet_v2_152/block3/unit_36/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                           num_features=1024,
                                                                                                           eps=1.0009999641624745e-05,
                                                                                                           momentum=0.0)
        self.resnet_v2_152_block3_unit_36_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_36/bottleneck_v2/conv1/Conv2D',
                                                                                   in_channels=1024, out_channels=256,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_36_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_36/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_36_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_36/bottleneck_v2/conv2/Conv2D',
                                                                                   in_channels=256, out_channels=256,
                                                                                   kernel_size=(3, 3), stride=(2, 2),
                                                                                   groups=1, bias=None)
        self.resnet_v2_152_block3_unit_36_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                    'resnet_v2_152/block3/unit_36/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                    num_features=256,
                                                                                                                    eps=1.0009999641624745e-05,
                                                                                                                    momentum=0.0)
        self.resnet_v2_152_block3_unit_36_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                   name='resnet_v2_152/block3/unit_36/bottleneck_v2/conv3/Conv2D',
                                                                                   in_channels=256, out_channels=1024,
                                                                                   kernel_size=(1, 1), stride=(1, 1),
                                                                                   groups=1, bias=True)
        self.resnet_v2_152_block4_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block4/unit_1/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=1024,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block4_unit_1_bottleneck_v2_shortcut_Conv2D = self.__conv(2,
                                                                                     name='resnet_v2_152/block4/unit_1/bottleneck_v2/shortcut/Conv2D',
                                                                                     in_channels=1024,
                                                                                     out_channels=2048,
                                                                                     kernel_size=(1, 1), stride=(1, 1),
                                                                                     groups=1, bias=True)
        self.resnet_v2_152_block4_unit_1_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block4/unit_1/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=1024, out_channels=512,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block4_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block4/unit_1/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=512,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block4_unit_1_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block4/unit_1/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=512, out_channels=512,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block4_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block4/unit_1/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=512,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block4_unit_1_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block4/unit_1/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=512, out_channels=2048,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block4_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block4/unit_2/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=2048,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block4_unit_2_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block4/unit_2/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=2048, out_channels=512,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block4_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block4/unit_2/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=512,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block4_unit_2_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block4/unit_2/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=512, out_channels=512,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block4_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block4/unit_2/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=512,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block4_unit_2_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block4/unit_2/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=512, out_channels=2048,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_block4_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                          'resnet_v2_152/block4/unit_3/bottleneck_v2/preact/FusedBatchNorm',
                                                                                                          num_features=2048,
                                                                                                          eps=1.0009999641624745e-05,
                                                                                                          momentum=0.0)
        self.resnet_v2_152_block4_unit_3_bottleneck_v2_conv1_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block4/unit_3/bottleneck_v2/conv1/Conv2D',
                                                                                  in_channels=2048, out_channels=512,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block4_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block4/unit_3/bottleneck_v2/conv1/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=512,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block4_unit_3_bottleneck_v2_conv2_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block4/unit_3/bottleneck_v2/conv2/Conv2D',
                                                                                  in_channels=512, out_channels=512,
                                                                                  kernel_size=(3, 3), stride=(1, 1),
                                                                                  groups=1, bias=None)
        self.resnet_v2_152_block4_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                                                   'resnet_v2_152/block4/unit_3/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm',
                                                                                                                   num_features=512,
                                                                                                                   eps=1.0009999641624745e-05,
                                                                                                                   momentum=0.0)
        self.resnet_v2_152_block4_unit_3_bottleneck_v2_conv3_Conv2D = self.__conv(2,
                                                                                  name='resnet_v2_152/block4/unit_3/bottleneck_v2/conv3/Conv2D',
                                                                                  in_channels=512, out_channels=2048,
                                                                                  kernel_size=(1, 1), stride=(1, 1),
                                                                                  groups=1, bias=True)
        self.resnet_v2_152_postnorm_FusedBatchNorm = self.__batch_normalization(2,
                                                                                'resnet_v2_152/postnorm/FusedBatchNorm',
                                                                                num_features=2048,
                                                                                eps=1.0009999641624745e-05,
                                                                                momentum=0.0)
        self.resnet_v2_152_logits_Conv2D = self.__conv(2, name='resnet_v2_152/logits/Conv2D', in_channels=2048,
                                                       out_channels=1001, kernel_size=(1, 1), stride=(1, 1), groups=1,
                                                       bias=True)

    def forward(self, x):
        resnet_v2_152_Pad = F.pad(x, (3, 3, 3, 3), mode='constant', value=0)
        resnet_v2_152_conv1_Conv2D = self.resnet_v2_152_conv1_Conv2D(resnet_v2_152_Pad)
        resnet_v2_152_pool1_MaxPool_pad = F.pad(resnet_v2_152_conv1_Conv2D, (0, 1, 0, 1), value=float('-inf'))
        resnet_v2_152_pool1_MaxPool, resnet_v2_152_pool1_MaxPool_idx = F.max_pool2d(resnet_v2_152_pool1_MaxPool_pad,
                                                                                    kernel_size=(3, 3), stride=(2, 2),
                                                                                    padding=0, ceil_mode=False,
                                                                                    return_indices=True)
        resnet_v2_152_block1_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block1_unit_1_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_pool1_MaxPool)
        resnet_v2_152_block1_unit_1_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block1_unit_1_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block1_unit_1_bottleneck_v2_shortcut_Conv2D = self.resnet_v2_152_block1_unit_1_bottleneck_v2_shortcut_Conv2D(
            resnet_v2_152_block1_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_152_block1_unit_1_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block1_unit_1_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block1_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_152_block1_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block1_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block1_unit_1_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block1_unit_1_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block1_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block1_unit_1_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block1_unit_1_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block1_unit_1_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block1_unit_1_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block1_unit_1_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block1_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block1_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block1_unit_1_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block1_unit_1_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block1_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block1_unit_1_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block1_unit_1_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block1_unit_1_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block1_unit_1_bottleneck_v2_add = resnet_v2_152_block1_unit_1_bottleneck_v2_shortcut_Conv2D + resnet_v2_152_block1_unit_1_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block1_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block1_unit_2_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block1_unit_1_bottleneck_v2_add)
        resnet_v2_152_block1_unit_2_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block1_unit_2_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block1_unit_2_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block1_unit_2_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block1_unit_2_bottleneck_v2_preact_Relu)
        resnet_v2_152_block1_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block1_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block1_unit_2_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block1_unit_2_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block1_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block1_unit_2_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block1_unit_2_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block1_unit_2_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block1_unit_2_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block1_unit_2_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block1_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block1_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block1_unit_2_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block1_unit_2_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block1_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block1_unit_2_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block1_unit_2_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block1_unit_2_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block1_unit_2_bottleneck_v2_add = resnet_v2_152_block1_unit_1_bottleneck_v2_add + resnet_v2_152_block1_unit_2_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block1_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block1_unit_3_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block1_unit_2_bottleneck_v2_add)
        resnet_v2_152_block1_unit_3_bottleneck_v2_shortcut_MaxPool, resnet_v2_152_block1_unit_3_bottleneck_v2_shortcut_MaxPool_idx = F.max_pool2d(
            resnet_v2_152_block1_unit_2_bottleneck_v2_add, kernel_size=(1, 1), stride=(2, 2), padding=0,
            ceil_mode=False, return_indices=True)
        resnet_v2_152_block1_unit_3_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block1_unit_3_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block1_unit_3_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block1_unit_3_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block1_unit_3_bottleneck_v2_preact_Relu)
        resnet_v2_152_block1_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block1_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block1_unit_3_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block1_unit_3_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block1_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block1_unit_3_bottleneck_v2_Pad = F.pad(resnet_v2_152_block1_unit_3_bottleneck_v2_conv1_Relu,
                                                              (1, 1, 1, 1), mode='constant', value=0)
        resnet_v2_152_block1_unit_3_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block1_unit_3_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block1_unit_3_bottleneck_v2_Pad)
        resnet_v2_152_block1_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block1_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block1_unit_3_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block1_unit_3_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block1_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block1_unit_3_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block1_unit_3_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block1_unit_3_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block1_unit_3_bottleneck_v2_add = resnet_v2_152_block1_unit_3_bottleneck_v2_shortcut_MaxPool + resnet_v2_152_block1_unit_3_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block2_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block2_unit_1_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block1_unit_3_bottleneck_v2_add)
        resnet_v2_152_block2_unit_1_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block2_unit_1_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block2_unit_1_bottleneck_v2_shortcut_Conv2D = self.resnet_v2_152_block2_unit_1_bottleneck_v2_shortcut_Conv2D(
            resnet_v2_152_block2_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_152_block2_unit_1_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block2_unit_1_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block2_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_152_block2_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_1_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block2_unit_1_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block2_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_1_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block2_unit_1_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block2_unit_1_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block2_unit_1_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block2_unit_1_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block2_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_1_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block2_unit_1_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block2_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_1_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block2_unit_1_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block2_unit_1_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block2_unit_1_bottleneck_v2_add = resnet_v2_152_block2_unit_1_bottleneck_v2_shortcut_Conv2D + resnet_v2_152_block2_unit_1_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block2_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block2_unit_2_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block2_unit_1_bottleneck_v2_add)
        resnet_v2_152_block2_unit_2_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block2_unit_2_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block2_unit_2_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block2_unit_2_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block2_unit_2_bottleneck_v2_preact_Relu)
        resnet_v2_152_block2_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_2_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block2_unit_2_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block2_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_2_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block2_unit_2_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block2_unit_2_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block2_unit_2_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block2_unit_2_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block2_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_2_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block2_unit_2_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block2_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_2_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block2_unit_2_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block2_unit_2_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block2_unit_2_bottleneck_v2_add = resnet_v2_152_block2_unit_1_bottleneck_v2_add + resnet_v2_152_block2_unit_2_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block2_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block2_unit_3_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block2_unit_2_bottleneck_v2_add)
        resnet_v2_152_block2_unit_3_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block2_unit_3_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block2_unit_3_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block2_unit_3_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block2_unit_3_bottleneck_v2_preact_Relu)
        resnet_v2_152_block2_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_3_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block2_unit_3_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block2_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_3_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block2_unit_3_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block2_unit_3_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block2_unit_3_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block2_unit_3_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block2_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_3_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block2_unit_3_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block2_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_3_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block2_unit_3_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block2_unit_3_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block2_unit_3_bottleneck_v2_add = resnet_v2_152_block2_unit_2_bottleneck_v2_add + resnet_v2_152_block2_unit_3_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block2_unit_4_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block2_unit_4_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block2_unit_3_bottleneck_v2_add)
        resnet_v2_152_block2_unit_4_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block2_unit_4_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block2_unit_4_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block2_unit_4_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block2_unit_4_bottleneck_v2_preact_Relu)
        resnet_v2_152_block2_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_4_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block2_unit_4_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block2_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_4_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block2_unit_4_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block2_unit_4_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block2_unit_4_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block2_unit_4_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block2_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_4_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block2_unit_4_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block2_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_4_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block2_unit_4_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block2_unit_4_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block2_unit_4_bottleneck_v2_add = resnet_v2_152_block2_unit_3_bottleneck_v2_add + resnet_v2_152_block2_unit_4_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block2_unit_5_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block2_unit_5_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block2_unit_4_bottleneck_v2_add)
        resnet_v2_152_block2_unit_5_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block2_unit_5_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block2_unit_5_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block2_unit_5_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block2_unit_5_bottleneck_v2_preact_Relu)
        resnet_v2_152_block2_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_5_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block2_unit_5_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block2_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_5_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block2_unit_5_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block2_unit_5_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block2_unit_5_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block2_unit_5_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block2_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_5_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block2_unit_5_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block2_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_5_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block2_unit_5_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block2_unit_5_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block2_unit_5_bottleneck_v2_add = resnet_v2_152_block2_unit_4_bottleneck_v2_add + resnet_v2_152_block2_unit_5_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block2_unit_6_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block2_unit_6_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block2_unit_5_bottleneck_v2_add)
        resnet_v2_152_block2_unit_6_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block2_unit_6_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block2_unit_6_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block2_unit_6_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block2_unit_6_bottleneck_v2_preact_Relu)
        resnet_v2_152_block2_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_6_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block2_unit_6_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block2_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_6_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block2_unit_6_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block2_unit_6_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block2_unit_6_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block2_unit_6_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block2_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_6_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block2_unit_6_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block2_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_6_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block2_unit_6_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block2_unit_6_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block2_unit_6_bottleneck_v2_add = resnet_v2_152_block2_unit_5_bottleneck_v2_add + resnet_v2_152_block2_unit_6_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block2_unit_7_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block2_unit_7_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block2_unit_6_bottleneck_v2_add)
        resnet_v2_152_block2_unit_7_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block2_unit_7_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block2_unit_7_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block2_unit_7_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block2_unit_7_bottleneck_v2_preact_Relu)
        resnet_v2_152_block2_unit_7_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_7_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_7_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block2_unit_7_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block2_unit_7_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_7_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block2_unit_7_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block2_unit_7_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block2_unit_7_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block2_unit_7_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block2_unit_7_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_7_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_7_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block2_unit_7_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block2_unit_7_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_7_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block2_unit_7_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block2_unit_7_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block2_unit_7_bottleneck_v2_add = resnet_v2_152_block2_unit_6_bottleneck_v2_add + resnet_v2_152_block2_unit_7_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block2_unit_8_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block2_unit_8_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block2_unit_7_bottleneck_v2_add)
        resnet_v2_152_block2_unit_8_bottleneck_v2_shortcut_MaxPool, resnet_v2_152_block2_unit_8_bottleneck_v2_shortcut_MaxPool_idx = F.max_pool2d(
            resnet_v2_152_block2_unit_7_bottleneck_v2_add, kernel_size=(1, 1), stride=(2, 2), padding=0,
            ceil_mode=False, return_indices=True)
        resnet_v2_152_block2_unit_8_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block2_unit_8_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block2_unit_8_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block2_unit_8_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block2_unit_8_bottleneck_v2_preact_Relu)
        resnet_v2_152_block2_unit_8_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_8_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_8_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block2_unit_8_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block2_unit_8_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_8_bottleneck_v2_Pad = F.pad(resnet_v2_152_block2_unit_8_bottleneck_v2_conv1_Relu,
                                                              (1, 1, 1, 1), mode='constant', value=0)
        resnet_v2_152_block2_unit_8_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block2_unit_8_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block2_unit_8_bottleneck_v2_Pad)
        resnet_v2_152_block2_unit_8_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block2_unit_8_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block2_unit_8_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block2_unit_8_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block2_unit_8_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block2_unit_8_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block2_unit_8_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block2_unit_8_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block2_unit_8_bottleneck_v2_add = resnet_v2_152_block2_unit_8_bottleneck_v2_shortcut_MaxPool + resnet_v2_152_block2_unit_8_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_1_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block2_unit_8_bottleneck_v2_add)
        resnet_v2_152_block3_unit_1_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_1_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_1_bottleneck_v2_shortcut_Conv2D = self.resnet_v2_152_block3_unit_1_bottleneck_v2_shortcut_Conv2D(
            resnet_v2_152_block3_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_1_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_1_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_1_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_1_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_1_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_1_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_1_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_1_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_1_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_1_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_1_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_1_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_1_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_1_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_1_bottleneck_v2_add = resnet_v2_152_block3_unit_1_bottleneck_v2_shortcut_Conv2D + resnet_v2_152_block3_unit_1_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_2_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_1_bottleneck_v2_add)
        resnet_v2_152_block3_unit_2_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_2_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_2_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_2_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_2_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_2_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_2_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_2_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_2_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_2_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_2_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_2_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_2_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_2_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_2_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_2_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_2_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_2_bottleneck_v2_add = resnet_v2_152_block3_unit_1_bottleneck_v2_add + resnet_v2_152_block3_unit_2_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_3_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_2_bottleneck_v2_add)
        resnet_v2_152_block3_unit_3_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_3_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_3_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_3_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_3_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_3_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_3_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_3_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_3_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_3_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_3_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_3_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_3_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_3_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_3_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_3_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_3_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_3_bottleneck_v2_add = resnet_v2_152_block3_unit_2_bottleneck_v2_add + resnet_v2_152_block3_unit_3_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_4_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_4_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_3_bottleneck_v2_add)
        resnet_v2_152_block3_unit_4_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_4_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_4_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_4_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_4_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_4_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_4_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_4_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_4_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_4_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_4_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_4_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_4_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_4_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_4_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_4_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_4_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_4_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_4_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_4_bottleneck_v2_add = resnet_v2_152_block3_unit_3_bottleneck_v2_add + resnet_v2_152_block3_unit_4_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_5_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_5_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_4_bottleneck_v2_add)
        resnet_v2_152_block3_unit_5_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_5_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_5_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_5_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_5_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_5_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_5_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_5_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_5_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_5_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_5_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_5_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_5_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_5_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_5_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_5_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_5_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_5_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_5_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_5_bottleneck_v2_add = resnet_v2_152_block3_unit_4_bottleneck_v2_add + resnet_v2_152_block3_unit_5_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_6_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_6_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_5_bottleneck_v2_add)
        resnet_v2_152_block3_unit_6_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_6_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_6_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_6_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_6_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_6_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_6_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_6_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_6_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_6_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_6_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_6_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_6_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_6_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_6_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_6_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_6_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_6_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_6_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_6_bottleneck_v2_add = resnet_v2_152_block3_unit_5_bottleneck_v2_add + resnet_v2_152_block3_unit_6_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_7_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_7_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_6_bottleneck_v2_add)
        resnet_v2_152_block3_unit_7_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_7_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_7_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_7_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_7_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_7_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_7_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_7_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_7_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_7_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_7_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_7_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_7_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_7_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_7_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_7_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_7_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_7_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_7_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_7_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_7_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_7_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_7_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_7_bottleneck_v2_add = resnet_v2_152_block3_unit_6_bottleneck_v2_add + resnet_v2_152_block3_unit_7_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_8_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_8_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_7_bottleneck_v2_add)
        resnet_v2_152_block3_unit_8_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_8_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_8_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_8_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_8_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_8_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_8_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_8_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_8_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_8_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_8_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_8_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_8_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_8_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_8_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_8_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_8_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_8_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_8_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_8_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_8_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_8_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_8_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_8_bottleneck_v2_add = resnet_v2_152_block3_unit_7_bottleneck_v2_add + resnet_v2_152_block3_unit_8_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_9_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_9_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_8_bottleneck_v2_add)
        resnet_v2_152_block3_unit_9_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_9_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_9_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_9_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_9_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_9_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_9_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_9_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_9_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_9_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_9_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_9_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_9_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_9_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_9_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_9_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_9_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_9_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_9_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_9_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_9_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_9_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_9_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_9_bottleneck_v2_add = resnet_v2_152_block3_unit_8_bottleneck_v2_add + resnet_v2_152_block3_unit_9_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_10_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_10_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_9_bottleneck_v2_add)
        resnet_v2_152_block3_unit_10_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_10_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_10_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_10_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_10_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_10_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_10_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_10_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_10_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_10_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_10_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_10_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_10_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_10_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_10_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_10_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_10_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_10_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_10_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_10_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_10_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_10_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_10_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_10_bottleneck_v2_add = resnet_v2_152_block3_unit_9_bottleneck_v2_add + resnet_v2_152_block3_unit_10_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_11_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_11_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_10_bottleneck_v2_add)
        resnet_v2_152_block3_unit_11_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_11_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_11_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_11_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_11_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_11_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_11_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_11_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_11_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_11_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_11_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_11_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_11_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_11_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_11_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_11_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_11_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_11_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_11_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_11_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_11_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_11_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_11_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_11_bottleneck_v2_add = resnet_v2_152_block3_unit_10_bottleneck_v2_add + resnet_v2_152_block3_unit_11_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_12_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_12_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_11_bottleneck_v2_add)
        resnet_v2_152_block3_unit_12_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_12_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_12_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_12_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_12_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_12_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_12_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_12_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_12_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_12_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_12_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_12_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_12_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_12_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_12_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_12_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_12_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_12_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_12_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_12_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_12_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_12_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_12_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_12_bottleneck_v2_add = resnet_v2_152_block3_unit_11_bottleneck_v2_add + resnet_v2_152_block3_unit_12_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_13_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_13_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_12_bottleneck_v2_add)
        resnet_v2_152_block3_unit_13_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_13_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_13_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_13_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_13_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_13_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_13_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_13_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_13_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_13_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_13_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_13_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_13_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_13_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_13_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_13_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_13_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_13_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_13_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_13_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_13_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_13_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_13_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_13_bottleneck_v2_add = resnet_v2_152_block3_unit_12_bottleneck_v2_add + resnet_v2_152_block3_unit_13_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_14_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_14_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_13_bottleneck_v2_add)
        resnet_v2_152_block3_unit_14_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_14_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_14_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_14_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_14_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_14_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_14_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_14_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_14_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_14_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_14_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_14_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_14_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_14_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_14_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_14_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_14_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_14_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_14_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_14_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_14_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_14_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_14_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_14_bottleneck_v2_add = resnet_v2_152_block3_unit_13_bottleneck_v2_add + resnet_v2_152_block3_unit_14_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_15_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_15_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_14_bottleneck_v2_add)
        resnet_v2_152_block3_unit_15_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_15_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_15_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_15_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_15_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_15_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_15_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_15_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_15_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_15_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_15_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_15_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_15_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_15_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_15_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_15_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_15_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_15_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_15_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_15_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_15_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_15_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_15_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_15_bottleneck_v2_add = resnet_v2_152_block3_unit_14_bottleneck_v2_add + resnet_v2_152_block3_unit_15_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_16_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_16_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_15_bottleneck_v2_add)
        resnet_v2_152_block3_unit_16_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_16_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_16_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_16_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_16_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_16_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_16_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_16_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_16_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_16_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_16_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_16_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_16_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_16_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_16_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_16_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_16_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_16_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_16_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_16_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_16_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_16_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_16_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_16_bottleneck_v2_add = resnet_v2_152_block3_unit_15_bottleneck_v2_add + resnet_v2_152_block3_unit_16_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_17_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_17_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_16_bottleneck_v2_add)
        resnet_v2_152_block3_unit_17_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_17_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_17_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_17_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_17_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_17_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_17_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_17_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_17_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_17_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_17_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_17_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_17_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_17_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_17_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_17_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_17_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_17_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_17_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_17_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_17_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_17_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_17_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_17_bottleneck_v2_add = resnet_v2_152_block3_unit_16_bottleneck_v2_add + resnet_v2_152_block3_unit_17_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_18_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_18_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_17_bottleneck_v2_add)
        resnet_v2_152_block3_unit_18_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_18_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_18_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_18_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_18_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_18_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_18_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_18_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_18_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_18_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_18_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_18_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_18_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_18_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_18_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_18_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_18_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_18_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_18_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_18_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_18_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_18_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_18_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_18_bottleneck_v2_add = resnet_v2_152_block3_unit_17_bottleneck_v2_add + resnet_v2_152_block3_unit_18_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_19_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_19_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_18_bottleneck_v2_add)
        resnet_v2_152_block3_unit_19_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_19_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_19_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_19_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_19_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_19_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_19_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_19_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_19_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_19_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_19_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_19_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_19_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_19_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_19_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_19_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_19_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_19_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_19_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_19_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_19_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_19_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_19_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_19_bottleneck_v2_add = resnet_v2_152_block3_unit_18_bottleneck_v2_add + resnet_v2_152_block3_unit_19_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_20_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_20_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_19_bottleneck_v2_add)
        resnet_v2_152_block3_unit_20_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_20_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_20_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_20_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_20_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_20_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_20_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_20_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_20_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_20_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_20_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_20_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_20_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_20_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_20_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_20_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_20_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_20_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_20_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_20_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_20_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_20_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_20_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_20_bottleneck_v2_add = resnet_v2_152_block3_unit_19_bottleneck_v2_add + resnet_v2_152_block3_unit_20_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_21_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_21_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_20_bottleneck_v2_add)
        resnet_v2_152_block3_unit_21_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_21_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_21_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_21_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_21_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_21_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_21_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_21_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_21_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_21_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_21_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_21_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_21_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_21_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_21_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_21_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_21_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_21_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_21_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_21_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_21_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_21_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_21_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_21_bottleneck_v2_add = resnet_v2_152_block3_unit_20_bottleneck_v2_add + resnet_v2_152_block3_unit_21_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_22_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_22_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_21_bottleneck_v2_add)
        resnet_v2_152_block3_unit_22_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_22_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_22_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_22_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_22_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_22_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_22_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_22_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_22_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_22_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_22_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_22_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_22_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_22_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_22_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_22_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_22_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_22_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_22_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_22_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_22_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_22_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_22_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_22_bottleneck_v2_add = resnet_v2_152_block3_unit_21_bottleneck_v2_add + resnet_v2_152_block3_unit_22_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_23_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_23_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_22_bottleneck_v2_add)
        resnet_v2_152_block3_unit_23_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_23_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_23_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_23_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_23_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_23_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_23_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_23_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_23_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_23_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_23_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_23_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_23_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_23_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_23_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_23_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_23_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_23_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_23_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_23_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_23_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_23_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_23_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_23_bottleneck_v2_add = resnet_v2_152_block3_unit_22_bottleneck_v2_add + resnet_v2_152_block3_unit_23_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_24_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_24_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_23_bottleneck_v2_add)
        resnet_v2_152_block3_unit_24_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_24_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_24_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_24_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_24_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_24_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_24_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_24_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_24_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_24_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_24_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_24_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_24_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_24_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_24_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_24_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_24_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_24_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_24_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_24_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_24_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_24_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_24_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_24_bottleneck_v2_add = resnet_v2_152_block3_unit_23_bottleneck_v2_add + resnet_v2_152_block3_unit_24_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_25_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_25_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_24_bottleneck_v2_add)
        resnet_v2_152_block3_unit_25_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_25_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_25_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_25_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_25_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_25_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_25_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_25_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_25_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_25_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_25_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_25_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_25_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_25_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_25_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_25_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_25_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_25_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_25_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_25_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_25_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_25_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_25_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_25_bottleneck_v2_add = resnet_v2_152_block3_unit_24_bottleneck_v2_add + resnet_v2_152_block3_unit_25_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_26_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_26_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_25_bottleneck_v2_add)
        resnet_v2_152_block3_unit_26_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_26_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_26_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_26_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_26_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_26_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_26_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_26_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_26_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_26_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_26_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_26_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_26_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_26_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_26_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_26_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_26_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_26_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_26_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_26_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_26_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_26_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_26_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_26_bottleneck_v2_add = resnet_v2_152_block3_unit_25_bottleneck_v2_add + resnet_v2_152_block3_unit_26_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_27_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_27_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_26_bottleneck_v2_add)
        resnet_v2_152_block3_unit_27_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_27_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_27_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_27_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_27_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_27_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_27_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_27_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_27_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_27_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_27_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_27_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_27_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_27_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_27_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_27_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_27_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_27_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_27_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_27_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_27_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_27_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_27_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_27_bottleneck_v2_add = resnet_v2_152_block3_unit_26_bottleneck_v2_add + resnet_v2_152_block3_unit_27_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_28_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_28_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_27_bottleneck_v2_add)
        resnet_v2_152_block3_unit_28_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_28_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_28_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_28_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_28_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_28_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_28_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_28_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_28_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_28_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_28_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_28_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_28_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_28_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_28_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_28_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_28_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_28_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_28_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_28_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_28_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_28_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_28_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_28_bottleneck_v2_add = resnet_v2_152_block3_unit_27_bottleneck_v2_add + resnet_v2_152_block3_unit_28_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_29_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_29_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_28_bottleneck_v2_add)
        resnet_v2_152_block3_unit_29_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_29_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_29_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_29_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_29_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_29_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_29_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_29_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_29_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_29_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_29_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_29_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_29_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_29_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_29_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_29_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_29_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_29_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_29_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_29_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_29_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_29_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_29_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_29_bottleneck_v2_add = resnet_v2_152_block3_unit_28_bottleneck_v2_add + resnet_v2_152_block3_unit_29_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_30_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_30_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_29_bottleneck_v2_add)
        resnet_v2_152_block3_unit_30_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_30_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_30_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_30_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_30_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_30_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_30_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_30_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_30_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_30_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_30_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_30_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_30_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_30_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_30_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_30_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_30_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_30_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_30_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_30_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_30_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_30_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_30_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_30_bottleneck_v2_add = resnet_v2_152_block3_unit_29_bottleneck_v2_add + resnet_v2_152_block3_unit_30_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_31_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_31_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_30_bottleneck_v2_add)
        resnet_v2_152_block3_unit_31_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_31_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_31_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_31_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_31_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_31_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_31_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_31_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_31_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_31_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_31_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_31_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_31_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_31_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_31_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_31_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_31_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_31_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_31_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_31_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_31_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_31_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_31_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_31_bottleneck_v2_add = resnet_v2_152_block3_unit_30_bottleneck_v2_add + resnet_v2_152_block3_unit_31_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_32_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_32_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_31_bottleneck_v2_add)
        resnet_v2_152_block3_unit_32_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_32_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_32_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_32_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_32_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_32_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_32_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_32_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_32_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_32_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_32_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_32_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_32_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_32_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_32_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_32_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_32_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_32_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_32_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_32_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_32_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_32_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_32_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_32_bottleneck_v2_add = resnet_v2_152_block3_unit_31_bottleneck_v2_add + resnet_v2_152_block3_unit_32_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_33_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_33_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_32_bottleneck_v2_add)
        resnet_v2_152_block3_unit_33_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_33_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_33_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_33_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_33_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_33_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_33_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_33_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_33_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_33_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_33_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_33_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_33_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_33_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_33_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_33_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_33_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_33_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_33_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_33_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_33_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_33_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_33_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_33_bottleneck_v2_add = resnet_v2_152_block3_unit_32_bottleneck_v2_add + resnet_v2_152_block3_unit_33_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_34_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_34_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_33_bottleneck_v2_add)
        resnet_v2_152_block3_unit_34_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_34_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_34_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_34_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_34_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_34_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_34_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_34_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_34_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_34_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_34_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_34_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_34_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_34_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_34_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_34_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_34_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_34_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_34_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_34_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_34_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_34_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_34_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_34_bottleneck_v2_add = resnet_v2_152_block3_unit_33_bottleneck_v2_add + resnet_v2_152_block3_unit_34_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_35_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_35_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_34_bottleneck_v2_add)
        resnet_v2_152_block3_unit_35_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_35_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_35_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_35_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_35_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_35_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_35_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_35_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_35_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_35_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_35_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block3_unit_35_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block3_unit_35_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_35_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_35_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block3_unit_35_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_35_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_35_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_35_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_35_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_35_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_35_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_35_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_35_bottleneck_v2_add = resnet_v2_152_block3_unit_34_bottleneck_v2_add + resnet_v2_152_block3_unit_35_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block3_unit_36_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block3_unit_36_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_35_bottleneck_v2_add)
        resnet_v2_152_block3_unit_36_bottleneck_v2_shortcut_MaxPool, resnet_v2_152_block3_unit_36_bottleneck_v2_shortcut_MaxPool_idx = F.max_pool2d(
            resnet_v2_152_block3_unit_35_bottleneck_v2_add, kernel_size=(1, 1), stride=(2, 2), padding=0,
            ceil_mode=False, return_indices=True)
        resnet_v2_152_block3_unit_36_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block3_unit_36_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block3_unit_36_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block3_unit_36_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block3_unit_36_bottleneck_v2_preact_Relu)
        resnet_v2_152_block3_unit_36_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_36_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_36_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block3_unit_36_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block3_unit_36_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_36_bottleneck_v2_Pad = F.pad(resnet_v2_152_block3_unit_36_bottleneck_v2_conv1_Relu,
                                                               (1, 1, 1, 1), mode='constant', value=0)
        resnet_v2_152_block3_unit_36_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block3_unit_36_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block3_unit_36_bottleneck_v2_Pad)
        resnet_v2_152_block3_unit_36_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block3_unit_36_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block3_unit_36_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block3_unit_36_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block3_unit_36_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block3_unit_36_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block3_unit_36_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block3_unit_36_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block3_unit_36_bottleneck_v2_add = resnet_v2_152_block3_unit_36_bottleneck_v2_shortcut_MaxPool + resnet_v2_152_block3_unit_36_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block4_unit_1_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block4_unit_1_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block3_unit_36_bottleneck_v2_add)
        resnet_v2_152_block4_unit_1_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block4_unit_1_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block4_unit_1_bottleneck_v2_shortcut_Conv2D = self.resnet_v2_152_block4_unit_1_bottleneck_v2_shortcut_Conv2D(
            resnet_v2_152_block4_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_152_block4_unit_1_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block4_unit_1_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block4_unit_1_bottleneck_v2_preact_Relu)
        resnet_v2_152_block4_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block4_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block4_unit_1_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block4_unit_1_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block4_unit_1_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block4_unit_1_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block4_unit_1_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block4_unit_1_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block4_unit_1_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block4_unit_1_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block4_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block4_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block4_unit_1_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block4_unit_1_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block4_unit_1_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block4_unit_1_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block4_unit_1_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block4_unit_1_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block4_unit_1_bottleneck_v2_add = resnet_v2_152_block4_unit_1_bottleneck_v2_shortcut_Conv2D + resnet_v2_152_block4_unit_1_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block4_unit_2_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block4_unit_2_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block4_unit_1_bottleneck_v2_add)
        resnet_v2_152_block4_unit_2_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block4_unit_2_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block4_unit_2_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block4_unit_2_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block4_unit_2_bottleneck_v2_preact_Relu)
        resnet_v2_152_block4_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block4_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block4_unit_2_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block4_unit_2_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block4_unit_2_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block4_unit_2_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block4_unit_2_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block4_unit_2_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block4_unit_2_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block4_unit_2_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block4_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block4_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block4_unit_2_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block4_unit_2_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block4_unit_2_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block4_unit_2_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block4_unit_2_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block4_unit_2_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block4_unit_2_bottleneck_v2_add = resnet_v2_152_block4_unit_1_bottleneck_v2_add + resnet_v2_152_block4_unit_2_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_block4_unit_3_bottleneck_v2_preact_FusedBatchNorm = self.resnet_v2_152_block4_unit_3_bottleneck_v2_preact_FusedBatchNorm(
            resnet_v2_152_block4_unit_2_bottleneck_v2_add)
        resnet_v2_152_block4_unit_3_bottleneck_v2_preact_Relu = F.relu(
            resnet_v2_152_block4_unit_3_bottleneck_v2_preact_FusedBatchNorm)
        resnet_v2_152_block4_unit_3_bottleneck_v2_conv1_Conv2D = self.resnet_v2_152_block4_unit_3_bottleneck_v2_conv1_Conv2D(
            resnet_v2_152_block4_unit_3_bottleneck_v2_preact_Relu)
        resnet_v2_152_block4_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block4_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block4_unit_3_bottleneck_v2_conv1_Conv2D)
        resnet_v2_152_block4_unit_3_bottleneck_v2_conv1_Relu = F.relu(
            resnet_v2_152_block4_unit_3_bottleneck_v2_conv1_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block4_unit_3_bottleneck_v2_conv2_Conv2D_pad = F.pad(
            resnet_v2_152_block4_unit_3_bottleneck_v2_conv1_Relu, (1, 1, 1, 1))
        resnet_v2_152_block4_unit_3_bottleneck_v2_conv2_Conv2D = self.resnet_v2_152_block4_unit_3_bottleneck_v2_conv2_Conv2D(
            resnet_v2_152_block4_unit_3_bottleneck_v2_conv2_Conv2D_pad)
        resnet_v2_152_block4_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm = self.resnet_v2_152_block4_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm(
            resnet_v2_152_block4_unit_3_bottleneck_v2_conv2_Conv2D)
        resnet_v2_152_block4_unit_3_bottleneck_v2_conv2_Relu = F.relu(
            resnet_v2_152_block4_unit_3_bottleneck_v2_conv2_BatchNorm_FusedBatchNorm)
        resnet_v2_152_block4_unit_3_bottleneck_v2_conv3_Conv2D = self.resnet_v2_152_block4_unit_3_bottleneck_v2_conv3_Conv2D(
            resnet_v2_152_block4_unit_3_bottleneck_v2_conv2_Relu)
        resnet_v2_152_block4_unit_3_bottleneck_v2_add = resnet_v2_152_block4_unit_2_bottleneck_v2_add + resnet_v2_152_block4_unit_3_bottleneck_v2_conv3_Conv2D
        resnet_v2_152_postnorm_FusedBatchNorm = self.resnet_v2_152_postnorm_FusedBatchNorm(
            resnet_v2_152_block4_unit_3_bottleneck_v2_add)
        resnet_v2_152_postnorm_Relu = F.relu(resnet_v2_152_postnorm_FusedBatchNorm)
        resnet_v2_152_pool5 = torch.mean(resnet_v2_152_postnorm_Relu, 3, True)
        resnet_v2_152_pool5 = torch.mean(resnet_v2_152_pool5, 2, True)
        resnet_v2_152_logits_Conv2D = self.resnet_v2_152_logits_Conv2D(resnet_v2_152_pool5)
        resnet_v2_152_SpatialSqueeze = torch.squeeze(resnet_v2_152_logits_Conv2D)
        MMdnn_Output_input = resnet_v2_152_SpatialSqueeze[:, 1:]
        return MMdnn_Output_input

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
