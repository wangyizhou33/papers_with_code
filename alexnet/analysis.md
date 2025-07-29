# AlexNet: Paper vs. PyTorch Implementation Analysis

This document analyzes the correspondences between the original AlexNet paper, "ImageNet Classification with Deep Convolutional Neural Networks," and a PyTorch implementation found in `./alexnet-pytorch/model.py`.

## 1. Architecture

The overall architecture described in the paper is largely followed by the implementation. The paper specifies 5 convolutional layers and 3 fully-connected layers.

### 1.1. Paper (Section 3.5)
- **Input:** 224x224x3 images.
- **Layer 1 (Conv):** 96 kernels of size 11x11x3 with a stride of 4.
- **Layer 2 (Conv):** 256 kernels of size 5x5x48.
- **Layer 3 (Conv):** 384 kernels of size 3x3x256.
- **Layer 4 (Conv):** 384 kernels of size 3x3x192.
- **Layer 5 (Conv):** 256 kernels of size 3x3x192.
- **Layer 6 (FC):** 4096 neurons.
- **Layer 7 (FC):** 4096 neurons.
- **Layer 8 (FC):** 1000 neurons (for the 1000 ImageNet classes).

### 1.2. Code (`model.py`, `AlexNet` class)
The `self.net` and `self.classifier` sequential models in the `AlexNet` class mirror this structure.

- **Input:** The code comments mention that the input size should be 227x227, not 224x224 as in the paper, to achieve the 55x55 feature map size after the first convolutional layer. This is a known discrepancy in the paper.
- **Convolutional Layers:** The `nn.Conv2d` layers in `self.net` match the kernel sizes, output channels, and strides described in the paper.
- **Fully-Connected Layers:** The `nn.Linear` layers in `self.classifier` match the number of neurons specified in the paper.

### 1.3. Parameter and Neuron Count
The claim of "60 million parameters and 650,000 neurons" is accurate. Here is a detailed breakdown based on the architecture described in the paper.

#### 1.3.1. Parameter Calculation
The formula for parameters in a convolutional layer is `(kernel_height * kernel_width * input_channels + 1) * num_kernels` (the +1 is for the bias). For a fully-connected layer, it's `(input_neurons + 1) * output_neurons`.

*   **Conv1:** `(11*11*3 + 1) * 96` = 34,944
*   **Conv2:** `(5*5*96 + 1) * 256` = 614,656
*   **Conv3:** `(3*3*256 + 1) * 384` = 885,120
*   **Conv4:** `(3*3*384 + 1) * 384` = 1,327,488
*   **Conv5:** `(3*3*384 + 1) * 256` = 884,992
*   **FC1 (Input: 6x6x256=9216):** `(9216 + 1) * 4096` = 37,748,736
*   **FC2:** `(4096 + 1) * 4096` = 16,781,312
*   **FC3:** `(4096 + 1) * 1000` = 4,097,000
*   **Total Parameters:** ~62.3 million (The commonly cited 60 million figure is a rounded value).

#### 1.3.2. Neuron Calculation
Neurons are the units in each layer. For convolutional layers, this is `output_height * output_width * num_kernels`.

*   **Conv1 (Output: 55x55):** `55 * 55 * 96` = 290,400
*   **Conv2 (Output: 27x27):** `27 * 27 * 256` = 186,624
*   **Conv3 (Output: 13x13):** `13 * 13 * 384` = 64,896
*   **Conv4 (Output: 13x13):** `13 * 13 * 384` = 64,896
*   **Conv5 (Output: 13x13):** `13 * 13 * 256` = 43,264
*   **FC1:** 4,096
*   **FC2:** 4,096
*   **FC3:** 1,000
*   **Total Neurons:** ~659,000 (The commonly cited 650,000 figure is a rounded value).

## 2. Key Features

### 2.1. ReLU Nonlinearity

**Paper (Section 3.1):** The paper emphasizes the use of Rectified Linear Units (ReLUs) for faster training compared to saturating activation functions like tanh or sigmoid.

**Code:** The implementation uses `nn.ReLU()` after each convolutional and fully-connected layer, which is consistent with the paper.

### 2.2. Local Response Normalization (LRN)

**Paper (Section 3.3):** The paper describes a local response normalization scheme to aid generalization. The formula is given as:
`b_{x,y}^i = a_{x,y}^i / (k + \alpha * \sum_{j=max(0, i-n/2)}^{min(N-1, i+n/2)} (a_{x,y}^j)^2)^\beta`
with `k=2`, `n=5`, `alpha=10^-4`, and `beta=0.75`.

**Code:** The implementation uses `nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)` after the first and second convolutional layers. This directly corresponds to the parameters in the paper.

### 2.3. Overlapping Pooling

**Paper (Section 3.4):** The paper uses overlapping max-pooling with a 3x3 window and a stride of 2.

**Code:** The implementation uses `nn.MaxPool2d(kernel_size=3, stride=2)` after the LRN layers and the fifth convolutional layer, which is consistent with the paper.

### 2.4. Dropout

**Paper (Section 4.2):** The paper applies dropout with a probability of 0.5 in the first two fully-connected layers to combat overfitting.

**Code:** The implementation uses `nn.Dropout(p=0.5, inplace=True)` in the `self.classifier` before the first two linear layers.

## 3. Training Details

### 3.1. Optimizer and Learning Rate

**Paper (Section 5):** The paper uses stochastic gradient descent (SGD) with a batch size of 128, momentum of 0.9, and weight decay of 0.0005. The learning rate was initialized at 0.01 and reduced by a factor of 10 when the validation error rate stopped improving.

**Code:** The implementation provides two options for the optimizer. The one that is commented out is the one that follows the paper's description:
```python
# optimizer = optim.SGD(
#     params=alexnet.parameters(),
#     lr=LR_INIT,
#     momentum=MOMENTUM,
#     weight_decay=LR_DECAY)
```
The code uses an `Adam` optimizer by default, with a comment stating that the SGD optimizer from the paper "doesn't train". This is a notable deviation from the paper. The learning rate scheduler, however, does follow the paper's strategy of reducing the learning rate.

### 3.2. Weight and Bias Initialization

**Paper (Section 5):** Weights are initialized from a zero-mean Gaussian distribution with a standard deviation of 0.01. Biases in the second, fourth, and fifth convolutional layers, as well as the fully-connected hidden layers, are initialized with the constant 1. Other biases are initialized with 0.

**Code:** The `init_bias` method in the `AlexNet` class implements this initialization scheme.
```python
def init_bias(self):
    for layer in self.net:
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            nn.init.constant_(layer.bias, 0)
    # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
    nn.init.constant_(self.net[4].bias, 1)
    nn.init.constant_(self.net[10].bias, 1)
    nn.init.constant_(self.net[12].bias, 1)
```

## 4. Data Augmentation

**Paper (Section 4.1):** The paper describes two forms of data augmentation:
1.  Extracting random 224x224 patches and their horizontal reflections from 256x256 images.
2.  Altering the intensities of RGB channels using PCA.

**Code:** The data loading and transformation section of the code includes `transforms.CenterCrop(IMAGE_DIM)` but has commented out `transforms.RandomResizedCrop` and `transforms.RandomHorizontalFlip`. This suggests that the data augmentation described in the paper is not fully implemented or used by default in this specific codebase.

## 5. Conclusion

The provided PyTorch implementation of AlexNet is a faithful reproduction of the architecture and most of the key features described in the paper. However, there are some notable differences, particularly in the choice of optimizer and the implementation of data augmentation. These differences are likely due to practical considerations in getting the model to train effectively.
