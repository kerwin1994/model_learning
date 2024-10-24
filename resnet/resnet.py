import torch
import torch.nn as nn

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, ##输入通道数
        out_planes, ## 输出通道数
        kernel_size=3, ## 卷积核大小
        stride=stride, ## 步长
        padding=dilation, ## 补丁 这样保证了即使使用dilation时也可以保持输入和输出的空间尺寸不变
        groups=groups, ## 输入和输出通道之间的对应关系
        bias=False, ## 偏置
        dilation=dilation, ## 空洞卷积膨胀系数
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    ##expansion：输出通道数与planes参数的比例。在BasicBlock中，这个值被固定为1，表示输出通道数等于planes。
    expansion: int = 1

    def __init__(
        self,
        inplanes: int, ## 输入的通道数
        planes: int, ## 中间层的输出通道数，由于expansion是1，最终输出通道也等于planes
        stride: int = 1, ## 步长
        ## 当输入输出大小不一样时，通过定义一个下采样模块也就是上面提到的1x1卷积层来改变输入大小
        downsample: Optional[nn.Module] = None, 
        groups: int = 1, ##只在BottleNeck中有用到，这里是固定值
        base_width: int = 64, ##只在BottleNeck中有用到，这里是固定值
        dilation: int = 1, ##只在BottleNeck中有用到，这里不支持空洞卷积
        norm_layer: Optional[Callable[..., nn.Module]] = None, ##规范化层，如果没定义就是BatchNorm2d
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64: 
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu( out)

        return out
    
class Bottleneck(nn.Module):
##在torchvision中，BottleNeck的下采样是在第二个卷积层（3x3）实现的，而原论文中是在第一个卷积层(1x1)实现的。这个改动对模型精度有提升。这个模型的变体通常被称为ResNet V1.5。
##其原理是，将下采样放置于3x3的卷积层而不是1x1的卷积层，使得模型在下采样之前，在更大的空间分辨率上操作。
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    ## expansion: Bottleneck块输出通道相对于planes参数的比例。
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64, ## 模块宽度的基准值，影响中间层的维度大小
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups ##计算中间层的通道数
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        ## stride应用在了conv2
        self.conv2 = conv3x3(width, width, stride, groups, dilation) 
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, BottleNeck]], ##残差块种类
        layers: List[int], ##每层残差块的数量
        num_classes: int = 1000, ##全连接层输出向量维度
        zero_init_residual: bool = False, ##具体在下面讲
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None, ##是否用空洞卷积代替stride，空洞卷积和stride都可以增大感受野，而空洞卷积可以在增大感受野同时保存特征图尺寸不变。在一些像素级应用（实例分割）很重要。
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ## 添加layers[0]个plane为64的block残差块
        self.layer1 = self._make_layer(block, 64, layers[0])
        ## 添加layers[1]个plane为128的block残差块,并且将尺寸减半
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        ##为每一层做权重初始化
        ##对卷积层做kaming初始化，也是本文的作者研究的针对带有relu激活函数的初始化方法。
        ##对规范化层，初始权重为1，偏置为0。这使得规范化层在开始训练时不对网络造成影响，让它在后面的训练中自己学习参数，这即简化了网络结构也避免了错误的权重初始化对训练造成影响。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        ## 通过将每个残差块的最后一个规范化层的权重设为0，使每个残差单元的输出接近于输入。这样一开始时就相当于模型所有的残差块都只做一个恒等映射。使模型的复杂度从简单到复杂慢慢提高，这有助于稳定训练过程，也能防止过拟合。
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        ## 用空洞卷积代替步长下采样
        if dilate:
            self.dilation *= stride ##特征尺寸不变，感受野与stride相等
            stride = 1
        ## 如果步长不等于1或者输入与输出通道数不等，输入需要做一个1x1的卷积映射
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        ## 将该残差块添加到网络
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        ## 第一个残差块的输出通道就是后面的残差块的输入通道数并且不会再变
        self.inplanes = planes * block.expansion
        ## 根据block数量重复添加该类型残差块，注意由于之前已经添加了一个，for循环从1开始
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)