# Semantic Segmentation and Classification Keras model warper 

## Get Started

The code is tested on python 3.6, tensorflow 1.4, Ubuntu 18.04.

Install packages for python3

```bash
sudo pip3 install tensorflow-gpu opencv-contrib-python matplotlib numpy keras
```

The weights are cached in server inside NTU campus, so you will not be able to access the files outside NTU.

```bash

# init temp path
mkdir -p tmp/weights

# deeplab
wget -P tmp/weights/deeplab http://ntu.h1fast.com/weights/deeplab/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5
wget -P tmp/weights/deeplab http://ntu.h1fast.com/weights/deeplab/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5
wget -P tmp/weights/deeplab http://ntu.h1fast.com/weights/deeplab/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5
wget -P tmp/weights/deeplab http://ntu.h1fast.com/weights/deeplab/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5

# xception
wget -P tmp/weights/xception http://ntu.h1fast.com/weights/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5
wget -P tmp/weights/xception http://ntu.h1fast.com/weights/xception/xception_weights_tf_dim_ordering_tf_kernels.h5

# inception
wget -P tmp/weights/inception http://ntu.h1fast.com/weights/inception/inception_v3_weights_tf_dim_ordering_tf_kernels.h5
wget -P tmp/weights/inception http://ntu.h1fast.com/weights/inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

# resnet
wget -P tmp/weights/resnet http://ntu.h1fast.com/weights/resnet_50/resnet50_weights_tf_dim_ordering_tf_kernels.h5
wget -P tmp/weights/resnet http://ntu.h1fast.com/weights/resnet_50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

# mobilenet
wget -P tmp/weights/mobilenet http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224.h5
wget -P tmp/weights/mobilenet http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5
wget -P tmp/weights/mobilenet http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5
wget -P tmp/weights/mobilenet http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_224_no_top.h5
wget -P tmp/weights/mobilenet http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top.h5
wget -P tmp/weights/mobilenet http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.3_224_no_top.h5
wget -P tmp/weights/mobilenet http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_224.h5
wget -P tmp/weights/mobilenet http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_224.h5
wget -P tmp/weights/mobilenet http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5
wget -P tmp/weights/mobilenet http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.3_224.h5
wget -P tmp/weights/mobilenet http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_224_no_top.h5
wget -P tmp/weights/mobilenet http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5

# inception_resnet
wget -P tmp/weights/inception_resnet http://ntu.h1fast.com/weights/inception_resnet/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5)
wget -P tmp/weights/inception_resnet http://ntu.h1fast.com/weights/inception_resnet/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5)

# densenet
wget -P tmp/weights/densenet http://ntu.h1fast.com/weights/densenet/densenet169_weights_tf_dim_ordering_tf_kernels.h5)
wget -P tmp/weights/densenet http://ntu.h1fast.com/weights/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5)
wget -P tmp/weights/densenet http://ntu.h1fast.com/weights/densenet/densenet121_weights_tf_dim_ordering_tf_kernels.h5)
wget -P tmp/weights/densenet http://ntu.h1fast.com/weights/densenet/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5)
wget -P tmp/weights/densenet http://ntu.h1fast.com/weights/densenet/densenet201_weights_tf_dim_ordering_tf_kernels.h5)
wget -P tmp/weights/densenet http://ntu.h1fast.com/weights/densenet/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5)

# nasnet
wget -P tmp/weights/nasnet http://ntu.h1fast.com/weights/nasnet/NASNet-mobile.h5)
wget -P tmp/weights/nasnet http://ntu.h1fast.com/weights/nasnet/NASNet-large-no-top.h5)
wget -P tmp/weights/nasnet http://ntu.h1fast.com/weights/nasnet/NASNet-large.h5)
wget -P tmp/weights/nasnet http://ntu.h1fast.com/weights/nasnet/NASNet-mobile-no-top.h5)
```