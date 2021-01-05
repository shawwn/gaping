from collections import namedtuple
from typing import Callable, Any, Optional, Tuple, List
import warnings
import gaping.tftorch as nn
import gaping.tftorch as F
import functools
Tensor = nn.Tensor

InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': Tensor, 'aux_logits': Optional[Tensor]}


class Inception3(nn.Module):

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
        init_weights: Optional[bool] = None,
        data_format: str = 'NHWC',
        scope='inceptionv3',
        **kwargs
    ) -> None:
        super(Inception3, self).__init__(scope=scope, **kwargs)
        with self.scope():
            self.data_format = data_format
            if inception_blocks is None:
                inception_blocks = [
                    BasicConv2d, InceptionA, InceptionB, InceptionC,
                    InceptionD, InceptionE, InceptionAux
                ]
            if init_weights is None:
                warnings.warn('The default weight initialization of inception_v3 will be changed in future releases of '
                              'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                              ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
                init_weights = True
            assert len(inception_blocks) == 7
            inception_blocks = [functools.partial(block, data_format=self.data_format)
                for block in inception_blocks]
            conv_block = inception_blocks[0]
            inception_a = inception_blocks[1]
            inception_b = inception_blocks[2]
            inception_c = inception_blocks[3]
            inception_d = inception_blocks[4]
            inception_e = inception_blocks[5]
            inception_aux = inception_blocks[6]

            self.aux_logits = aux_logits
            self.transform_input = transform_input
            self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2, scope='Conv2d_1a_3x3')
            self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3, scope='Conv2d_2a_3x3')
            self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1, scope='Conv2d_2b_3x3')
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, scope='maxpool1')
            self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1, scope='Conv2d_3b_1x1')
            self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3, scope='Conv2d_4a_3x3')
            self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, scope='maxpool2')
            self.Mixed_5b = inception_a(192, pool_features=32, scope='Mixed_5b')
            self.Mixed_5c = inception_a(256, pool_features=64, scope='Mixed_5c')
            self.Mixed_5d = inception_a(288, pool_features=64, scope='Mixed_5d')
            self.Mixed_6a = inception_b(288, scope='Mixed_6a')
            self.Mixed_6b = inception_c(768, channels_7x7=128, scope='Mixed_6b')
            self.Mixed_6c = inception_c(768, channels_7x7=160, scope='Mixed_6c')
            self.Mixed_6d = inception_c(768, channels_7x7=160, scope='Mixed_6d')
            self.Mixed_6e = inception_c(768, channels_7x7=192, scope='Mixed_6e')
            self.AuxLogits: Optional[nn.Module] = None
            if aux_logits:
                self.AuxLogits = inception_aux(768, num_classes, scope='aux_logits')
            self.Mixed_7a = inception_d(768, scope='Mixed_7a')
            self.Mixed_7b = inception_e(1280, scope='Mixed_7b')
            self.Mixed_7c = inception_e(2048, scope='Mixed_7c')
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1), scope='avgpool')
            self.dropout = nn.Dropout()
            self.fc = nn.Linear(2048, num_classes, scope='fc', weight_name='weight', bias_name='bias')
            # if init_weights:
            #     for m in self.modules():
            #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            #             import scipy.stats as stats
            #             stddev = m.stddev if hasattr(m, 'stddev') else 0.1
            #             X = stats.truncnorm(-2, 2, scale=stddev)
            #             values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
            #             values = values.view(m.weight.size())
            #             with torch.no_grad():
            #                 m.weight.copy_(values)
            #         elif isinstance(m, nn.BatchNorm2d):
            #             nn.init.constant_(m.weight, 1)
            #             nn.init.constant_(m.bias, 0)

    def _forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling.
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = nn.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> InceptionOutputs:
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x  # type: ignore[return-value]

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = nn.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = nn.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = nn.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = nn.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x: Tensor) -> InceptionOutputs:
        with self.scope():
            x = self._transform_input(x)
            x, aux = self._forward(x)
            aux_defined = self.training and self.aux_logits
            if False and torch.jit.is_scripting():
                if not aux_defined:
                    warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
                return InceptionOutputs(x, aux)
            else:
                return self.eager_outputs(x, aux)


class InceptionBlock(nn.Module):
    def __init__(self, data_format = 'NHWC', scope: str = None):
        if scope is None:
            raise ValueError('Must specify scope')
        super(InceptionBlock, self).__init__(scope=scope)
        self.data_format = data_format
        self.channel_axis = 1 if self.data_format == 'NCHW' else 3

    def forward(self, x: Tensor) -> Tensor:
        with self.scope():
            outputs = self._forward(x)
            return nn.cat(outputs, self.channel_axis)


class InceptionA(InceptionBlock):
    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        scope: str = None,
        **kws
    ) -> None:
        super(InceptionA, self).__init__(scope=scope, **kws)
        with self.scope():
            if conv_block is None:
                conv_block = BasicConv2d
            self.branch1x1 = conv_block(in_channels, 64, kernel_size=1, scope='branch1x1')

            self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1, scope='branch5x5_1')
            self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2, scope='branch5x5_2')

            self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1, scope='branch3x3dbl_1')
            self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1, scope='branch3x3dbl_2')
            self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1, scope='branch3x3dbl_3')

            self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1, scope='branch_pool')

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs




class InceptionB(InceptionBlock):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        scope: str = None,
        **kws
    ) -> None:
        super(InceptionB, self).__init__(scope=scope, **kws)
        with self.scope():
            if conv_block is None:
                conv_block = BasicConv2d
            self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2, scope='branch3x3')

            self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1, scope='branch3x3dbl_1')
            self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1, scope='branch3x3dbl_2')
            self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2, scope='branch3x3dbl_3')

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs


class InceptionC(InceptionBlock):

    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        scope: str = None,
        **kws
    ) -> None:
        super(InceptionC, self).__init__(scope=scope, **kws)
        with self.scope():
            if conv_block is None:
                conv_block = BasicConv2d
            self.branch1x1 = conv_block(in_channels, 192, kernel_size=1, scope='branch1x1')

            c7 = channels_7x7
            self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1, scope='branch7x7_1')
            self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3), scope='branch7x7_2')
            self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0), scope='branch7x7_3')

            self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1, scope='branch7x7dbl_1')
            self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0), scope='branch7x7dbl_2')
            self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3), scope='branch7x7dbl_3')
            self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0), scope='branch7x7dbl_4')
            self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3), scope='branch7x7dbl_5')

            self.branch_pool = conv_block(in_channels, 192, kernel_size=1, scope='branch_pool')

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs


class InceptionD(InceptionBlock):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        scope: str = None,
        **kws
    ) -> None:
        super(InceptionD, self).__init__(scope=scope, **kws)
        with self.scope():
            if conv_block is None:
                conv_block = BasicConv2d
            self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1, scope='branch3x3_1')
            self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2, scope='branch3x3_2')

            self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1, scope='branch7x7x3_1')
            self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3), scope='branch7x7x3_2')
            self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0), scope='branch7x7x3_3')
            self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2, scope='branch7x7x3_4')

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs


class InceptionE(InceptionBlock):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        scope: str = None,
        **kws
    ) -> None:
        super(InceptionE, self).__init__(scope=scope, **kws)
        with self.scope():
            if conv_block is None:
                conv_block = BasicConv2d
            self.branch1x1 = conv_block(in_channels, 320, kernel_size=1, scope='branch1x1')

            self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1, scope='branch3x3_1')
            self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1), scope='branch3x3_2a')
            self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0), scope='branch3x3_2b')

            self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1, scope='branch3x3dbl_1')
            self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1, scope='branch3x3dbl_2')
            self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1), scope='branch3x3dbl_3a')
            self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0), scope='branch3x3dbl_3b')

            self.branch_pool = conv_block(in_channels, 192, kernel_size=1, scope='branch_pool')

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = nn.cat(branch3x3, self.channel_axis)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = nn.cat(branch3x3dbl, self.channel_axis)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs


class InceptionAux(InceptionBlock):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        scope: str = None,
        **kws
    ) -> None:
        super(InceptionAux, self).__init__(scope=scope, **kws)
        with self.scope():
            if conv_block is None:
                conv_block = BasicConv2d
            self.conv0 = conv_block(in_channels, 128, kernel_size=1, scope='conv0')
            self.conv1 = conv_block(128, 768, kernel_size=5, scope='conv1')
            self.conv1.stddev = 0.01  # TODO: support this
            self.fc = nn.Linear(768, num_classes, scope='fc')
            self.fc.stddev = 0.001  # TODO: support this

    def forward(self, x: Tensor) -> Tensor:
        with self.scope():
            # N x 768 x 17 x 17
            x = F.avg_pool2d(x, kernel_size=5, stride=3)
            # N x 768 x 5 x 5
            x = self.conv0(x)
            # N x 128 x 5 x 5
            x = self.conv1(x)
            # N x 768 x 1 x 1
            # Adaptive average pooling
            x = F.adaptive_avg_pool2d(x, (1, 1))
            # N x 768 x 1 x 1
            x = nn.flatten(x, 1)
            # N x 768
            x = self.fc(x)
            # N x 1000
            return x



class BasicConv2d(InceptionBlock):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data_format: str = 'NHWC',
        scope='conv',
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__(scope=scope, data_format=data_format)
        with self.scope():
            self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs, scope='conv', weight_name='weight', bias_name='bias')
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001, scope='bn')

    def forward(self, x: Tensor) -> Tensor:
        with self.scope():
            x = self.conv(x)
            x = self.bn(x)
            return F.relu(x, inplace=True)

