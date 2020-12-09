import torch
import torch.nn as nn
from . import functional

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(
                inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels *
                BasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(
                out_channels *
                BasicBlock.expansion))

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels *
                    BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(
                    out_channels *
                    BasicBlock.expansion))

    def forward(self, x):
        return nn.ReLU(
            inplace=True)(
            self.residual_function(x) +
            self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(
                inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                stride=stride,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(
                inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels *
                BottleNeck.expansion,
                kernel_size=1,
                bias=False),
            nn.BatchNorm2d(
                out_channels *
                BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels *
                    BottleNeck.expansion,
                    stride=stride,
                    kernel_size=1,
                    bias=False),
                nn.BatchNorm2d(
                    out_channels *
                    BottleNeck.expansion))

    def forward(self, x):
        return nn.ReLU(
            inplace=True)(
            self.residual_function(x) +
            self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron network layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        logits = self.fc(output)
        return logits


class ResNet_fc(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron network layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        last_fc = output.view(output.size(0), -1)
        return last_fc


def resnet18(num_classes):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet18_fc(num_classes):
    """
    Args:
        num_classes: numclasses

    Returns:
        ResNet18 object
    """
    return ResNet_fc(BasicBlock, [2,2,2,2], num_classes=num_classes)


def resnet34(num_classes):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet34_fc(num_classes):
    """ return a ResNet 34 object
    """
    return ResNet_fc(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)

class ensemble_3_resnet18(nn.Module):

    def __init__(self, num_classes,num_models=3, resnet18=resnet18):
        assert num_models == 3
        super(ensemble_3_resnet18, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet18(self.num_classes)
        self.model_2 = resnet18(self.num_classes)
        self.model_3 = resnet18(self.num_classes)

    def forward(self, x):
        logit_1 = self.model_1(x)
        logit_2 = self.model_2(x)
        logit_3 = self.model_3(x)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)

        probs = torch.cat([prob_1, prob_2, prob_3],dim=-1)
        return probs

class avg_ensemble_3_resnet18(nn.Module):

    def __init__(self, num_classes, num_models=3, resnet18=resnet18):
        assert num_models == 3
        super(avg_ensemble_3_resnet18, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet18(self.num_classes)
        self.model_2 = resnet18(self.num_classes)
        self.model_3 = resnet18(self.num_classes)

    def forward(self, x):
        logit_1 = self.model_1(x)
        logit_2 = self.model_2(x)
        logit_3 = self.model_3(x)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)

        avg_probs = (prob_1+prob_2+prob_3) / self.num_models
        return avg_probs

class ensemble_3_resnet18_fc(nn.Module):
    
    def __init__(self, num_classes, num_models=3, resnet18_fc=resnet18_fc):
        assert num_models == 3
        super(ensemble_3_resnet18_fc, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet18_fc(self.num_classes)
        self.model_2 = resnet18_fc(self.num_classes)
        self.model_3 = resnet18_fc(self.num_classes)
        self.fc1 = nn.Linear(512, self.num_classes)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.fc3 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        fc_1 = self.model_1(x)
        fc_2 = self.model_2(x)
        fc_3 = self.model_3(x)

        fcs = torch.cat([fc_1, fc_2, fc_3], dim=-1)        
        fcs = functional.dropout_parallel(fcs, self.num_models, 10, 0.9, 0.1, training=self.training)
        fcs_1, fcs_2, fcs_3 = torch.split(fcs, 512, dim=1)

        logit_1 = self.fc1(fcs_1)
        logit_2 = self.fc2(fcs_2)
        logit_3 = self.fc3(fcs_3)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)

        probs = torch.cat([prob_1, prob_2, prob_3], dim=-1)
        return probs

class avg_ensemble_3_resnet18_fc(nn.Module):

    def __init__(self, num_classes, num_models=3, resnet18_fc=resnet18_fc):
        assert num_models == 3
        super(avg_ensemble_3_resnet18_fc, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet18_fc(self.num_classes)
        self.model_2 = resnet18_fc(self.num_classes)
        self.model_3 = resnet18_fc(self.num_classes)
        self.fc1 = nn.Linear(512, self.num_classes)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.fc3 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        fc_1 = self.model_1(x)
        fc_2 = self.model_2(x)
        fc_3 = self.model_3(x)

        logit_1 = self.fc1(fc_1)
        logit_2 = self.fc2(fc_2)
        logit_3 = self.fc3(fc_3)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)

        avg_probs = (prob_1+prob_2+prob_3) / self.num_models
        return avg_probs

class ensemble_5_resnet18(nn.Module):

    def __init__(self, num_classes,num_models=5,resnet18=resnet18):
        assert num_models == 5
        super(ensemble_5_resnet18, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet18(self.num_classes)
        self.model_2 = resnet18(self.num_classes)
        self.model_3 = resnet18(self.num_classes)
        self.model_4 = resnet18(self.num_classes)
        self.model_5 = resnet18(self.num_classes)

    def forward(self, x):
        logit_1 = self.model_1(x)
        logit_2 = self.model_2(x)
        logit_3 = self.model_3(x)
        logit_4 = self.model_4(x)
        logit_5 = self.model_5(x)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)
        prob_4 = nn.Softmax(dim=1)(logit_4)
        prob_5 = nn.Softmax(dim=1)(logit_5)

        probs = torch.cat([prob_1, prob_2, prob_3, prob_4, prob_5],dim=-1)
        return probs

class avg_ensemble_5_resnet18(nn.Module):

    def __init__(self, num_classes, num_models=5, resnet18=resnet18):
        assert num_models == 5
        super(avg_ensemble_5_resnet18, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet18(self.num_classes)
        self.model_2 = resnet18(self.num_classes)
        self.model_3 = resnet18(self.num_classes)
        self.model_4 = resnet18(self.num_classes)
        self.model_5 = resnet18(self.num_classes)

    def forward(self, x):
        logit_1 = self.model_1(x)
        logit_2 = self.model_2(x)
        logit_3 = self.model_3(x)
        logit_4 = self.model_4(x)
        logit_5 = self.model_5(x)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)
        prob_4 = nn.Softmax(dim=1)(logit_4)
        prob_5 = nn.Softmax(dim=1)(logit_5)

        avg_probs = (prob_1+prob_2+prob_3+prob_4+prob_5) / self.num_models
        return avg_probs

class ensemble_5_resnet18_fc(nn.Module):

    def __init__(self, num_classes,num_models=5,resnet18_fc=resnet18_fc):
        assert num_models == 5
        super(ensemble_5_resnet18_fc, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet18_fc(self.num_classes)
        self.model_2 = resnet18_fc(self.num_classes)
        self.model_3 = resnet18_fc(self.num_classes)
        self.model_4 = resnet18_fc(self.num_classes)
        self.model_5 = resnet18_fc(self.num_classes)
        self.fc1 = nn.Linear(512, self.num_classes)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.fc3 = nn.Linear(512, self.num_classes)
        self.fc4 = nn.Linear(512, self.num_classes)
        self.fc5 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        fc_1 = self.model_1(x)
        fc_2 = self.model_2(x)
        fc_3 = self.model_3(x)
        fc_4 = self.model_4(x)
        fc_5 = self.model_5(x)
        fcs = torch.cat([fc_1, fc_2, fc_3, fc_4, fc_5], dim=-1)
        fcs = functional.dropout_parallel(fcs, self.num_models, 20, 0.9, 0.1, training=self.training)
        fcs_1, fcs_2, fcs_3, fcs_4, fcs_5 = torch.split(fcs, 512, dim=1)
        
        logit_1 = self.fc1(fcs_1)
        logit_2 = self.fc2(fcs_2)
        logit_3 = self.fc3(fcs_3)
        logit_4 = self.fc4(fcs_4)
        logit_5 = self.fc5(fcs_5)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)
        prob_4 = nn.Softmax(dim=1)(logit_4)
        prob_5 = nn.Softmax(dim=1)(logit_5)

        probs = torch.cat([prob_1, prob_2, prob_3, prob_4, prob_5], dim=-1)
        return probs

class avg_ensemble_5_resnet18_fc(nn.Module):

    def __init__(self, num_classes, num_models=5, resnet18_fc=resnet18_fc):
        assert num_models == 5
        super().__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet18_fc(self.num_classes)
        self.model_2 = resnet18_fc(self.num_classes)
        self.model_3 = resnet18_fc(self.num_classes)
        self.model_4 = resnet18_fc(self.num_classes)
        self.model_5 = resnet18_fc(self.num_classes)
        self.fc1 = nn.Linear(512, self.num_classes)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.fc3 = nn.Linear(512, self.num_classes)
        self.fc4 = nn.Linear(512, self.num_classes)
        self.fc5 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        fc_1 = self.model_1(x)
        fc_2 = self.model_2(x)
        fc_3 = self.model_3(x)
        fc_4 = self.model_4(x)
        fc_5 = self.model_5(x)

        logit_1 = self.fc1(fc_1)
        logit_2 = self.fc2(fc_2)
        logit_3 = self.fc3(fc_3)
        logit_4 = self.fc4(fc_4)
        logit_5 = self.fc5(fc_5)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)
        prob_4 = nn.Softmax(dim=1)(logit_4)
        prob_5 = nn.Softmax(dim=1)(logit_5)

        avg_probs = (prob_1+prob_2+prob_3+prob_4+prob_5) / self.num_models
        return avg_probs

class ensemble_3_resnet34(nn.Module):

    def __init__(self, num_classes,num_models=3,resnet34=resnet34):
        assert num_models == 3
        super(ensemble_3_resnet34, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet34(self.num_classes)
        self.model_2 = resnet34(self.num_classes)
        self.model_3 = resnet34(self.num_classes)

    def forward(self, x):
        logit_1 = self.model_1(x)
        logit_2 = self.model_2(x)
        logit_3 = self.model_3(x)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)

        probs = torch.cat([prob_1, prob_2, prob_3],dim=-1)
        return probs

class avg_ensemble_3_resnet34(nn.Module):

    def __init__(self, num_classes, num_models=3, resnet34=resnet34):
        assert num_models == 3
        super(avg_ensemble_3_resnet34, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet34(self.num_classes)
        self.model_2 = resnet34(self.num_classes)
        self.model_3 = resnet34(self.num_classes)


    def forward(self, x):
        logit_1 = self.model_1(x)
        logit_2 = self.model_2(x)
        logit_3 = self.model_3(x)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)

        avg_probs = (prob_1+prob_2+prob_3) / self.num_models
        return avg_probs

class ensemble_3_resnet34_fc(nn.Module):
    
    def __init__(self, num_classes, num_models=3, resnet18_fc=resnet34_fc):
        assert num_models == 3
        super(ensemble_3_resnet34_fc, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet18_fc(self.num_classes)
        self.model_2 = resnet18_fc(self.num_classes)
        self.model_3 = resnet18_fc(self.num_classes)
        self.fc1 = nn.Linear(512, self.num_classes)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.fc3 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        fc_1 = self.model_1(x)
        fc_2 = self.model_2(x)
        fc_3 = self.model_3(x)

        fcs = torch.cat([fc_1, fc_2, fc_3], dim=-1)        
        fcs = functional.dropout_parallel(fcs, self.num_models, 10, 0.9, 0.1, training=self.training)
        fcs_1, fcs_2, fcs_3 = torch.split(fcs, 512, dim=1)

        logit_1 = self.fc1(fcs_1)
        logit_2 = self.fc2(fcs_2)
        logit_3 = self.fc3(fcs_3)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)

        probs = torch.cat([prob_1, prob_2, prob_3], dim=-1)
        return probs

class avg_ensemble_3_resnet34_fc(nn.Module):

    def __init__(self, num_classes, num_models=3, resnet18_fc=resnet34_fc):
        assert num_models == 3
        super(avg_ensemble_3_resnet34_fc, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet18_fc(self.num_classes)
        self.model_2 = resnet18_fc(self.num_classes)
        self.model_3 = resnet18_fc(self.num_classes)
        self.fc1 = nn.Linear(512, self.num_classes)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.fc3 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        fc_1 = self.model_1(x)
        fc_2 = self.model_2(x)
        fc_3 = self.model_3(x)

        logit_1 = self.fc1(fc_1)
        logit_2 = self.fc2(fc_2)
        logit_3 = self.fc3(fc_3)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)

        avg_probs = (prob_1+prob_2+prob_3) / self.num_models
        return avg_probs

class ensemble_5_resnet34(nn.Module):

    def __init__(self, num_classes,num_models=5,resnet34=resnet34):
        assert num_models == 5
        super(ensemble_5_resnet34, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet34(self.num_classes)
        self.model_2 = resnet34(self.num_classes)
        self.model_3 = resnet34(self.num_classes)
        self.model_4 = resnet34(self.num_classes)
        self.model_5 = resnet34(self.num_classes)

    def forward(self, x):
        logit_1 = self.model_1(x)
        logit_2 = self.model_2(x)
        logit_3 = self.model_3(x)
        logit_4 = self.model_4(x)
        logit_5 = self.model_5(x)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)
        prob_4 = nn.Softmax(dim=1)(logit_4)
        prob_5 = nn.Softmax(dim=1)(logit_5)

        probs = torch.cat([prob_1, prob_2, prob_3, prob_4, prob_5],dim=-1)
        return probs

class avg_ensemble_5_resnet34(nn.Module):

    def __init__(self, num_classes, num_models=5, resnet34=resnet34):
        assert num_models == 5
        super(avg_ensemble_5_resnet34, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet34(self.num_classes)
        self.model_2 = resnet34(self.num_classes)
        self.model_3 = resnet34(self.num_classes)
        self.model_4 = resnet34(self.num_classes)
        self.model_5 = resnet34(self.num_classes)

    def forward(self, x):
        logit_1 = self.model_1(x)
        logit_2 = self.model_2(x)
        logit_3 = self.model_3(x)
        logit_4 = self.model_4(x)
        logit_5 = self.model_5(x)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)
        prob_4 = nn.Softmax(dim=1)(logit_4)
        prob_5 = nn.Softmax(dim=1)(logit_5)

        avg_probs = (prob_1+prob_2+prob_3+prob_4+prob_5) / self.num_models
        return avg_probs

class ensemble_5_resnet34_fc(nn.Module):

    def __init__(self, num_classes,num_models=5,resnet34_fc=resnet34_fc):
        assert num_models == 5
        super(ensemble_5_resnet34_fc, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet34_fc(self.num_classes)
        self.model_2 = resnet34_fc(self.num_classes)
        self.model_3 = resnet34_fc(self.num_classes)
        self.model_4 = resnet34_fc(self.num_classes)
        self.model_5 = resnet34_fc(self.num_classes)
        self.fc1 = nn.Linear(512, self.num_classes)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.fc3 = nn.Linear(512, self.num_classes)
        self.fc4 = nn.Linear(512, self.num_classes)
        self.fc5 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        fc_1 = self.model_1(x)
        fc_2 = self.model_2(x)
        fc_3 = self.model_3(x)
        fc_4 = self.model_4(x)
        fc_5 = self.model_5(x)

        fcs = torch.cat([fc_1, fc_2, fc_3, fc_4, fc_5], dim=-1)
        fcs = functional.dropout_parallel(fcs, self.num_models, 20, 0.9, 0.1, training=self.training)
        fcs_1, fcs_2, fcs_3, fcs_4, fcs_5 = torch.split(fcs, 512, dim=1)
        
        logit_1 = self.fc1(fcs_1)
        logit_2 = self.fc2(fcs_2)
        logit_3 = self.fc3(fcs_3)
        logit_4 = self.fc4(fcs_4)
        logit_5 = self.fc5(fcs_5)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)
        prob_4 = nn.Softmax(dim=1)(logit_4)
        prob_5 = nn.Softmax(dim=1)(logit_5)

        probs = torch.cat([prob_1, prob_2, prob_3, prob_4, prob_5], dim=-1)
        return probs

class avg_ensemble_5_resnet34_fc(nn.Module):

    def __init__(self, num_classes, num_models=5, resnet34_fc=resnet34_fc):
        assert num_models == 5
        super(avg_ensemble_5_resnet34_fc, self).__init__()
        self.num_classes = num_classes
        self.num_models = num_models
        self.model_1 = resnet34_fc(self.num_classes)
        self.model_2 = resnet34_fc(self.num_classes)
        self.model_3 = resnet34_fc(self.num_classes)
        self.model_4 = resnet34_fc(self.num_classes)
        self.model_5 = resnet34_fc(self.num_classes)
        self.fc1 = nn.Linear(512, self.num_classes)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.fc3 = nn.Linear(512, self.num_classes)
        self.fc4 = nn.Linear(512, self.num_classes)
        self.fc5 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        fc_1 = self.model_1(x)
        fc_2 = self.model_2(x)
        fc_3 = self.model_3(x)
        fc_4 = self.model_4(x)
        fc_5 = self.model_5(x)

        logit_1 = self.fc1(fc_1)
        logit_2 = self.fc2(fc_2)
        logit_3 = self.fc3(fc_3)
        logit_4 = self.fc4(fc_4)
        logit_5 = self.fc5(fc_5)

        prob_1 = nn.Softmax(dim=1)(logit_1)
        prob_2 = nn.Softmax(dim=1)(logit_2)
        prob_3 = nn.Softmax(dim=1)(logit_3)
        prob_4 = nn.Softmax(dim=1)(logit_4)
        prob_5 = nn.Softmax(dim=1)(logit_5)

        avg_probs = (prob_1+prob_2+prob_3+prob_4+prob_5) / self.num_models
        return avg_probs