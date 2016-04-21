# Convolutional Neural Network for texture classification


#### Architecture description

We used a 12-layered CNN, composed of a series of conv->relu->pool layers followed by a conv->relu->softmax design. In this way, the first convolution layer was composed of 32 3x3 filters, a number based on the number of filters the filter bank designed by Malik had. We followed this with a rectified linear unit and pooling layer with stride 2. We repeated this pattern 2 more times, increasing the number of filters in each "repeat" from 32, to 64 to 128 (multiples of 2) reducing our volume to a 14x14x128 volume, which we convolved with 25 14x14 filters to reduce the output to 1x1x25. 

We used this architecture design based on the VGGNet architecture that was provided in the CNN practical by Andrea Vedaldi. We noticed that this net was organized in conv-relu-pooling units and decided to try the same thing, although with less layers to take into account the constraint of 1 hour of training time. On a shared virtual machine of 64 GB RAM and 1.5 TB of hard disk space, it took our network 14.000 seconds (i.e 3.8 hours) to train. This was due to the number of epochs we used, 5: 1 epoch took less than 1 hour to complete. 

### Results

Shown below are the confusion matrix of predictions and objective and error functions in terms of run epochs.

![Figure 1](https://github.com/milongo/lab_vision/blob/master/lab8_cnn/figure3.png)
![Figure 2](https://github.com/milongo/lab_vision/blob/master/lab8_cnn/net-train.png)

As can be seen in the images above, classification was extremely poor. Perhaps a better design of the CNN architecture could be attempted. 

