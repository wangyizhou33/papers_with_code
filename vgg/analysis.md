# VGG Network: Paper to Code Correspondence

This document highlights the direct correspondence between the theoretical VGG network architecture described in the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" by Simonyan and Zisserman (2014) and its PyTorch implementation found in `model.py` within the `VGG-PyTorch` repository.

## 1. Introduction to VGG

The VGG network is a seminal convolutional neural network architecture known for its simplicity and effectiveness. Its key characteristic is the use of very small (3x3) convolutional filters throughout the entire network, emphasizing depth over width. The paper explores various configurations of VGG, varying in depth from 11 to 19 weight layers.

## 2. Architecture Overview

The VGG paper describes a general architecture consisting of:
- Stacks of convolutional layers with 3x3 filters, stride 1, and 1-pixel padding.
- Max-pooling layers with 2x2 windows and stride 2.
- Three fully-connected layers at the end: two with 4096 channels each, and the final one with 1000 channels (for ILSVRC classification).
- ReLU activation after every convolutional and fully-connected layer.
- Optional Local Response Normalisation (LRN) and Batch Normalisation (BN).

## 3. Configuration Mapping (`vgg_cfgs` in `model.py`)

The `model.py` file defines a dictionary `vgg_cfgs` that directly maps to the different VGG configurations (VGG11, VGG13, VGG16, VGG19) presented in Table 1 of the paper. Each list within the dictionary represents the sequence of convolutional layer output channels and 'M' for max-pooling operations.

```python
vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
```

This `vgg_cfgs` dictionary precisely defines the number of convolutional layers and their output depths, as well as the placement of max-pooling layers, mirroring Table 1 in the paper.

## 4. Convolutional Layers (`_make_layers` function)

The `_make_layers` function in `model.py` is responsible for constructing the convolutional blocks.

```python
def _make_layers(vgg_cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: nn.Sequential[nn.Module] = nn.Sequential()
    in_channels = 3
    for v in vgg_cfg:
        if v == "M":
            layers.append(nn.MaxPool2d((2, 2), (2, 2)))
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1)) # 3x3 kernel, stride 1, padding 1
            if batch_norm:
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(True))
            in_channels = v
    return layers
```

- **3x3 Filters**: The `nn.Conv2d` layer is initialized with `(3, 3)` for the kernel size, directly matching the paper's emphasis on small convolutional filters.
- **Stride 1, Padding 1**: The `(1, 1)` for stride and padding ensures that the spatial resolution is preserved after convolution, as stated in Section 2.1 of the paper.
- **ReLU Activation**: `nn.ReLU(True)` is appended after each convolutional layer (and batch normalization if used), consistent with the paper's mention that "All hidden layers are equipped with the rectification (ReLU) non-linearity."
- **Batch Normalization**: The `batch_norm` parameter allows for the inclusion of `nn.BatchNorm2d`, which corresponds to the "vggXX_bn" configurations in the paper.

## 5. Max-Pooling Layers

Within the `_make_layers` function, when 'M' is encountered in the `vgg_cfg`, a `nn.MaxPool2d` layer is added:

```python
            if v == "M":
                layers.append(nn.MaxPool2d((2, 2), (2, 2)))
```

This directly implements the max-pooling operation described in the paper: "Max-pooling is performed over a 2 × 2 pixel window, with stride 2."

## 6. Fully-Connected Layers (`classifier` module)

The `VGG` class's `__init__` method defines the `classifier` module:

```python
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
```

- **4096-4096-1000 Structure**: This sequential module precisely matches the paper's description of the fully-connected layers: "the first two have 4096 channels each, the third performs 1000-way ILSVRC classification and thus contains 1000 channels (one for each class)." The `num_classes` parameter allows for flexibility beyond 1000.
- **ReLU Activation**: `nn.ReLU(True)` is applied after each linear layer, consistent with the paper.
- **Dropout**: `nn.Dropout(0.5)` is included after the first two fully-connected layers, matching the paper's mention of "dropout regularisation for the first two fully-connected layers (dropout ratio set to 0.5)."
- **Adaptive Average Pooling**: Before the classifier, `self.avgpool = nn.AdaptiveAvgPool2d((7, 7))` is used. This prepares the feature maps for the fixed-size input of the fully-connected layers, effectively replacing the dense application and spatial averaging mentioned in the paper's testing section (Section 3.2). The paper states: "the fully-connected layers are first converted to convolutional layers (the first FC layer to a 7 × 7 conv. layer, the last two FC layers to 1 × 1 conv. layers). The resulting fully-convolutional net is then applied to the whole (uncropped) image. The result is a class score map with the number of channels equal to the number of classes, and a variable spatial resolution, dependent on the input image size. Finally, to obtain a fixed-size vector of class scores for the image, the class score map is spatially averaged (sum-pooled)." The `AdaptiveAvgPool2d((7,7))` achieves this spatial averaging to produce a 7x7 output, which is then flattened.

## 7. Weight Initialization (`_initialize_weights` method)

The `_initialize_weights` method in `model.py` implements the weight initialization strategy:

```python
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
```

The paper states in Section 3.1, "For random initialisation (where applicable), we sampled the weights from a normal distribution with the zero mean and 10-2 variance. The biases were initialised with zero."
- For `nn.Linear` layers, `nn.init.normal_(module.weight, 0, 0.01)` directly corresponds to this, as 0.01 is 10^-2.
- For `nn.Conv2d` layers, `nn.init.kaiming_normal_` is used, which is a common and effective initialization strategy for layers followed by ReLU, and is a more modern approach than simple normal distribution for convolutional layers. The biases are initialized to 0, as stated in the paper.
- For `nn.BatchNorm2d` layers, weights are initialized to 1 and biases to 0, which are standard practices for batch normalization layers.

## 8. Forward Pass (`_forward_impl` method)

The `_forward_impl` method outlines the data flow through the network:

```python
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
```

This clearly shows the sequential processing: input `x` goes through the `features` (convolutional and pooling layers), then `avgpool` (adaptive average pooling), then `flatten` to prepare for the `classifier` (fully-connected layers). This matches the overall flow of the VGG architecture.

## 9. Conclusion

The PyTorch implementation in `model.py` demonstrates a strong and faithful adherence to the VGG architecture described in the paper. From the specific configurations of VGG11 to VGG19, the use of 3x3 convolutional filters, 2x2 max-pooling, the structure of the fully-connected layers, and even the weight initialization strategies, the code accurately reflects the theoretical design. This close correspondence makes the repository an excellent resource for understanding the practical implementation of VGG networks.
