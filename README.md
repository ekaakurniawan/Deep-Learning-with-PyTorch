# Deep Learning with PyTorch

## Benchmark Result

### Project-0
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
 - CPU Type: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
 - CPU Cores: 6
 - CPU Threads: 12
 - Memory: 16 GiB
 - GPU Type: Nvidia GeForce GTX 1060
 - GPU Memory: 6 GiB
 
__Software Version__
 - Python: 3.6.7
 - PyTorch: 0.4.1.post2
 - TorchVision: 0.2.1
 - NumPy: 1.15.4

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
