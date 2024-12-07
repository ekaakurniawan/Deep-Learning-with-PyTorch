# Deep Learning with PyTorch

## Contents

### Introduction to Neural Networks
1. Gradient Descent Algorithm
   - [Implementing the Gradient Descent Algorithm](intro-neural-networks/gradient-descent/GradientDescent.ipynb)
2. Neural Networks Applications
   - [Predicting Student Admissions](intro-neural-networks/student-admissions/StudentAdmissions.ipynb)

### Introduction to PyTorch
- [Part 1 - Tensors in PyTorch](intro-to-pytorch/Part%201%20-%20Tensors%20in%20PyTorch%20(Exercises).ipynb)
- [Part 2 - Neural Networks in PyTorch](intro-to-pytorch/Part%202%20-%20Neural%20Networks%20in%20PyTorch%20(Exercises).ipynb)
- [Part 3 - Training Neural Networks](intro-to-pytorch/Part%203%20-%20Training%20Neural%20Networks%20(Exercises).ipynb)
- [Part 4 - Fashion-MNIST](intro-to-pytorch/Part%204%20-%20Fashion-MNIST%20(Exercises).ipynb)
- [Part 5 - Inference and Validation](intro-to-pytorch/Part%205%20-%20Inference%20and%20Validation%20(Exercises).ipynb)
- [Part 6 - Saving and Loading Models](intro-to-pytorch/Part%206%20-%20Saving%20and%20Loading%20Models.ipynb)
- [Part 7 - Loading Image Data](intro-to-pytorch/Part%207%20-%20Loading%20Image%20Data%20(Exercises).ipynb)
- [Part 8 - Transfer Learning](intro-to-pytorch/Part%208%20-%20Transfer%20Learning%20(Exercises).ipynb)

### Convolutional Neural Networks
1. Image Classification using Multi-Layer Perceptron
   - [Image Classification using Multi-Layer Perceptron and MNIST Images](convolutional-neural-networks/mnist-mlp/mnist_mlp_exercise.ipynb)
2. Image Classification using Convolutional Neural Networks
   - [Load and Augment CIFAR-10 Images](convolutional-neural-networks/cifar-cnn/cifar10_cnn_augmentation.ipynb)
   - [Image Classification using Convolutional Neural Networks and CIFAR-10 Images](convolutional-neural-networks/cifar-cnn/cifar10_cnn_exercise.ipynb)
3. Convolutional Visualization
   - [Convolutional Layer](convolutional-neural-networks/conv-visualization/conv_visualization.ipynb)
   - [Creating a Filter, Edge Detection](convolutional-neural-networks/conv-visualization/custom_filters.ipynb)
   - [Maxpooling Layer](convolutional-neural-networks/conv-visualization/maxpooling_visualization.ipynb)

### Style Transfer
- [Style Transfer with Deep Neural Networks](style-transfer/Style_Transfer_Exercise.ipynb)
- [Style Transfer Slides, AI Guild, Bukalapak, 29 March 2019](./slides/DL-Style-Transfer-Slides.pdf)

### Recurrent Neural Networks
1. Recurrent Neural Networks Applications
   - [Simple RNN for Time-Series Prediction](recurrent-neural-networks/time-series/Simple_RNN.ipynb)
2. LSTM Applications
   - [Character-Level LSTM to Generate New Text](recurrent-neural-networks/char-rnn/Character_Level_RNN_Exercise.ipynb)

### Sentiment Analysis
- [Sentiment Analysis with an RNN](sentiment-rnn/Sentiment_RNN_Exercise.ipynb)

### Project-0: Image Classification to Recognize Different Species of Flowers
1. [VGG16, SGD, woNorm](project-0/1_vgg16_sgd_woNorm/Image%20Classifier%20Project.ipynb)
2. [VGG16, SGD, wNorm](project-0/2_vgg16_sgd_wNorm/Image%20Classifier%20Project.ipynb)
3. [VGG16, SGD, wNorm, Momentum](project-0/3_vgg16_sgd_wNorm_momentum/Image%20Classifier%20Project.ipynb)
4. [VGG16, Adam, wNorm](project-0/4_vgg16_adam_wNorm/Image%20Classifier%20Project.ipynb)
5. [InceptionV3, SGD, wNorm](project-0/5_inceptionV3_sgd_wNorm/Image%20Classifier%20Project.ipynb)
6. [ResNet152, SGD, wNorm](project-0/6_resnet152_sgd_wNorm/Image%20Classifier%20Project.ipynb)
7. [DenseNet121, SGD, wNorm, Momentum, wFineTuning](project-0/7_train_densenet121_sgd_wNorm_momentum_contGrad_predNoRand/Image%20Classifier%20Project.ipynb)

## Setup

### Intel GPU

Please follow 
[Getting Started on Intel GPU](https://pytorch.org/docs/stable/notes/get_start_xpu.html)
article from PyTorch.

__Note__: If your system has an NPU, please disable it. One method is by 
revoking the access from the current user group. Change NPU group access from 
render to root.

To revoke NPU access from the current user group.
```
$ sudo chgrp root /dev/accel/accel0 
```

To grant NPU access back to the current user group.
```
$ sudo chgrp render /dev/accel/accel0 
```

Tested on the following hardware specification and software version.

__Hardware Specification__
 - CPU: Intel® Core™ Ultra 9 Processor 285K
 - CPU Cores: 24 (8 Performance-cores and 16 Efficient-cores)
 - CPU Threads: 24
 - Memory: 32 GiB
 - GPU: Intel® Arc™ A770 Graphics
 - GPU Memory: 16 GiB
 
__Software Version__
 - Ubuntu 24.04.1 LTS
 - Python 3.12.2
 - PyTorch 2.5.1+xpu
 - TorchVision 0.20.1+xpu
 - NumPy 1.26.3
 - Matplotlib 0.1.7
 - Pandas 2.2.3
 - intel-for-pytorch-gpu-dev-0.5
 - intel-pti-dev-0.9

### Install Requirements

Create virtual environment.
```
$ python3 -m venv pytorch_arc_env
$ source pytorch_arc_env/bin/activate
$ python -m pip install --upgrade pip
```

Install PyTorch and other required packages.
```
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/xpu
$ pip install --upgrade -r requirements.txt
```

### Test Run

Activate virtual environment and setup variables.
```
$ source pytorch_arc_env/bin/activate
$ source /opt/intel/oneapi/pytorch-gpu-dev-0.5/oneapi-vars.sh
$ source /opt/intel/oneapi/pti/0.9/env/vars.sh
```

Detect GPU.
```
$ python -c "import torch; print(torch.xpu.is_available())"
```
```
True
```

Run notebooks.
```
$ cd Deep-Learning-with-PyTorch
$ jupyter lab
```

## Monitoring Tools

### Ubuntu
 - [top](https://man7.org/linux/man-pages/man1/top.1.html): CPU utilization.
 - [Intel GPU top](https://manpages.ubuntu.com/manpages/noble/man1/intel_gpu_top.1.html): Intel iGPU and dGPU utilization.
 - [Intel PCM](https://github.com/intel/pcm): Intel CPU and iGPU power consumption.
 - [Intel XPU-SMI](https://intel.github.io/xpumanager/smi_install_guide.html): Intel dGPU power consumption.
 - [NVIDIA SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html): NVIDIA dGPU utilization and power consumption.

### Windows

 - [HWiNFO](https://www.hwinfo.com/): CPU, iGPU, and dGPU utilization and power consumption.

## Benchmark Result

### Project-0: Image Classification to Recognize Different Species of Flowers
__Terms__
 - NB: Notebook
 - Norm: Input Image Normalization

__Data__
 - Total Training Images: 6552
 - Total Testing Images: 818

__Hyperparameters__
 - Mntm (Momentum): 0.9
 - Learning Rate: 0.001
 
__Hardware Specification__
 - CPU Type: Intel® Core™ i7-8750H Processor 
 - CPU Cores: 6
 - CPU Threads: 12
 - Memory: 16 GiB
 - GPU Type: NVIDIA GeForce® GTX 1060
 - GPU Memory: 6 GiB
 
__Software Version__
 - Python 3.6.7
 - PyTorch 0.4.1.post2
 - TorchVision 0.2.1
 - NumPy 1.15.4
 - CUDA 9.0

__Column Info__
  - Training Time is for the total of training images
  - Testing Time is for the total of testing images

```
------------------------------------------------------------------------------------------------------------
NB   Model          Optimizer   Norm   GPU Utilization (%)   GPU Memory    Training   Testing   Best Testing
                                       Training   Testing    Consumption   Time       Time      Accuracy
                                                             (MiB)         (s)        (s)       (%)
------------------------------------------------------------------------------------------------------------
1    VGG16          SGD         No       -          -           -           -          -        78
2    VGG16          SGD         Yes    100          -           -          69          8        88
3    VGG16          SGD+Mntm    Yes    100         98        1945          71          7        92
4    VGG16          Adam        Yes    100          -        4096           -          -        65
5    Inception V3   SGD         Yes     97         97        1228          47          6        82
6    ResNet152      SGD         Yes    100        100        1024          73         10        86
7    DenseNet121    SGD+Mntm    Yes     98         98         999          35          4        95
------------------------------------------------------------------------------------------------------------
```
