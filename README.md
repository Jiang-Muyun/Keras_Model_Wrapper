# Semantic Segmentation and Classification Keras model warper 

## Get Started

The code is tested on python 3.6, tensorflow 1.4, Ubuntu 18.04.

Install packages for python3

```bash
sudo pip3 install tensorflow-gpu opencv-contrib-python matplotlib numpy keras
```

Init workspace, and download weights.

```bash
mkdir -p tmp/weights
cd tmp/weights
mkdir deeplab inception mobilenet resnet_50 xception inception_resnet densenet nasnet
cd deeplab
wget http://ntu.h1fast.com/weights/deeplab/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5
```

## All weights download

The weights are cached in server inside NTU campus, so you will not be able to access the files outside NTU. Direct download very large h5 file from Github can be really slow, our cache server can provide more than 50 times faster download speed than Github.

### deeplab

- [deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5](http://ntu.h1fast.com/weights/deeplab/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5)
- [deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5](http://ntu.h1fast.com/weights/deeplab/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5)
- [deeplabv3_xception_tf_dim_ordering_tf_kernels.h5](http://ntu.h1fast.com/weights/deeplab/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5)
- [deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5](http://ntu.h1fast.com/weights/deeplab/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5)

### xception

- [xception_weights_tf_dim_ordering_tf_kernels_notop.h5](http://ntu.h1fast.com/weights/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5)
- [xception_weights_tf_dim_ordering_tf_kernels.h5](http://ntu.h1fast.com/weights/xception/xception_weights_tf_dim_ordering_tf_kernels.h5)

### inception

- [inception_v3_weights_tf_dim_ordering_tf_kernels.h5](http://ntu.h1fast.com/weights/inception/inception_v3_weights_tf_dim_ordering_tf_kernels.h5)
- [inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5](http://ntu.h1fast.com/weights/inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5)

### resnet

- [resnet50_weights_tf_dim_ordering_tf_kernels.h5](http://ntu.h1fast.com/weights/resnet_50/resnet50_weights_tf_dim_ordering_tf_kernels.h5)
- [resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5](http://ntu.h1fast.com/weights/resnet_50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5)

### mobilenet

- [mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224.h5](http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224.h5)
- [mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5](http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5)
- [mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5](http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5)
- [mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_224_no_top.h5](http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_224_no_top.h5)
- [mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top.h5](http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top.h5)
- [mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.3_224_no_top.h5](http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.3_224_no_top.h5)
- [mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_224.h5](http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_224.h5)
- [mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_224.h5](http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_224.h5)
- [mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5](http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5)
- [mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.3_224.h5](http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.3_224.h5)
- [mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_224_no_top.h5](http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_224_no_top.h5)
- [mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5](http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5)

### inception_resnet

- [inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5](http://ntu.h1fast.com/weights/inception_resnet/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5)
- [inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5](http://ntu.h1fast.com/weights/inception_resnet/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5)

### densenet

- [densenet169_weights_tf_dim_ordering_tf_kernels.h5](http://ntu.h1fast.com/weights/densenet/densenet169_weights_tf_dim_ordering_tf_kernels.h5)
- [densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5](http://ntu.h1fast.com/weights/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5)
- [densenet121_weights_tf_dim_ordering_tf_kernels.h5](http://ntu.h1fast.com/weights/densenet/densenet121_weights_tf_dim_ordering_tf_kernels.h5)
- [densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5](http://ntu.h1fast.com/weights/densenet/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5)
- [densenet201_weights_tf_dim_ordering_tf_kernels.h5](http://ntu.h1fast.com/weights/densenet/densenet201_weights_tf_dim_ordering_tf_kernels.h5)
- [densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5](http://ntu.h1fast.com/weights/densenet/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5)

### nasnet

- [NASNet-mobile.h5](http://ntu.h1fast.com/weights/nasnet/NASNet-mobile.h5)
- [NASNet-large-no-top.h5](http://ntu.h1fast.com/weights/nasnet/NASNet-large-no-top.h5)
- [NASNet-large.h5](http://ntu.h1fast.com/weights/nasnet/NASNet-large.h5)
- [NASNet-mobile-no-top.h5](http://ntu.h1fast.com/weights/nasnet/NASNet-mobile-no-top.h5)
