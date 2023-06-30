# Creating a CIFAR 10 Convolution Neural Network

    This repo contains a CNN for training CIFAR 10 Dataset.

`model.py` file contains the CNN Model. It has `FinalModel` which is the final lighter model with under 50k parameters and produces over 70% train and test accuracy for the CIFAR 10 Dataset.

Model Summary is as Follows -

```
Device Type - mps
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 64, 32, 32]          18,432
              ReLU-6           [-1, 64, 32, 32]               0
       BatchNorm2d-7           [-1, 64, 32, 32]             128
           Dropout-8           [-1, 64, 32, 32]               0
            Conv2d-9           [-1, 16, 32, 32]           1,040
           Conv2d-10           [-1, 16, 28, 28]           2,304
             ReLU-11           [-1, 16, 28, 28]               0
      BatchNorm2d-12           [-1, 16, 28, 28]              32
           Conv2d-13           [-1, 32, 28, 28]           4,608
             ReLU-14           [-1, 32, 28, 28]               0
      BatchNorm2d-15           [-1, 32, 28, 28]              64
          Dropout-16           [-1, 32, 28, 28]               0
           Conv2d-17           [-1, 64, 28, 28]          18,432
             ReLU-18           [-1, 64, 28, 28]               0
      BatchNorm2d-19           [-1, 64, 28, 28]             128
          Dropout-20           [-1, 64, 28, 28]               0
           Conv2d-21           [-1, 32, 28, 28]           2,080
           Conv2d-22           [-1, 32, 24, 24]           9,216
             ReLU-23           [-1, 32, 24, 24]               0
      BatchNorm2d-24           [-1, 32, 24, 24]              64
           Conv2d-25           [-1, 64, 24, 24]          18,432
             ReLU-26           [-1, 64, 24, 24]               0
      BatchNorm2d-27           [-1, 64, 24, 24]             128
          Dropout-28           [-1, 64, 24, 24]               0
           Conv2d-29           [-1, 64, 24, 24]             576
           Conv2d-30          [-1, 256, 24, 24]          16,384
             ReLU-31          [-1, 256, 24, 24]               0
      BatchNorm2d-32          [-1, 256, 24, 24]             512
          Dropout-33          [-1, 256, 24, 24]               0
           Conv2d-34           [-1, 32, 24, 24]           8,224
           Conv2d-35           [-1, 32, 20, 20]           9,216
             ReLU-36           [-1, 32, 20, 20]               0
      BatchNorm2d-37           [-1, 32, 20, 20]              64
           Conv2d-38           [-1, 64, 20, 20]          18,432
             ReLU-39           [-1, 64, 20, 20]               0
      BatchNorm2d-40           [-1, 64, 20, 20]             128
          Dropout-41           [-1, 64, 20, 20]               0
           Conv2d-42           [-1, 64, 20, 20]             576
           Conv2d-43          [-1, 256, 20, 20]          16,384
             ReLU-44          [-1, 256, 20, 20]               0
      BatchNorm2d-45          [-1, 256, 20, 20]             512
          Dropout-46          [-1, 256, 20, 20]               0
        AvgPool2d-47            [-1, 256, 1, 1]               0
           Linear-48                   [-1, 10]           2,560
================================================================
Total params: 149,584
Trainable params: 149,584
Non-trainable params: 0
----------------------------------------------------------------
```

Model Accuracy:
- Training Accuracy: 81.63
- Test Accuracy: 86.00


Incorrect Predictions Images:

![Incorrect Predictions](<CleanShot 2023-06-29 at 16.14.10@2x.png>)