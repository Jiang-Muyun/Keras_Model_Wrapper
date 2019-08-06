# Segmentation and classification simple warpper

The code is tested on python 3.6, tensorflow 1.14.0, Keras 2.2.4, Ubuntu 18.04. [Demo Video](https://www.youtube.com/watch?v=UnnYx1wMz68)

- Segmentation models supported
    + DeepLabv3 Xception
    + DeepLabv3 MobileNetV2 
    + Mask_RCNN ResNet50
- Classification models supported
    + Xception
    + InceptionV3
    + ResNet50
    + MobileNetV2
    + VGG16, VGG19
- Models that keras.application supported
    + Inception_resnet
    + Densenet
    + NASNet

## Get Started

The weights can only be accessible inside NTU campus.

```bash
# install python dependences
pip3 install -r requirements.txt

# Download the files that you need
# airplane test videos
wget -P tmp/videos/ http://ntu.h1fast.com/airplane/9_Very_Close_Takeoffs_Landings.mp4
wget -P tmp/videos/ http://ntu.h1fast.com/airplane/20_Landings_in_9_Minutes.mp4
wget -P tmp/videos/ http://ntu.h1fast.com/airplane/Landing_with_strong_side_wind.mp4

# deeplab weights
wget -P tmp/weights/deeplab/ http://ntu.h1fast.com/weights/deeplab/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5
wget -P tmp/weights/deeplab/ http://ntu.h1fast.com/weights/deeplab/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5
wget -P tmp/weights/deeplab/ http://ntu.h1fast.com/weights/deeplab/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5
wget -P tmp/weights/deeplab/ http://ntu.h1fast.com/weights/deeplab/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5

# mask_rcnn
wget -P tmp/weights/mask_rcnn http://ntu.h1fast.com/weights/mask_rcnn/mask_rcnn_coco.h5

# xception weights
wget -P tmp/weights/xception/ http://ntu.h1fast.com/weights/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5
wget -P tmp/weights/xception/ http://ntu.h1fast.com/weights/xception/xception_weights_tf_dim_ordering_tf_kernels.h5

# inception weights
wget -P tmp/weights/inception/ http://ntu.h1fast.com/weights/inception/inception_v3_weights_tf_dim_ordering_tf_kernels.h5
wget -P tmp/weights/inception/ http://ntu.h1fast.com/weights/inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

# resnet weights
wget -P tmp/weights/resnet/ http://ntu.h1fast.com/weights/resnet_50/resnet50_weights_tf_dim_ordering_tf_kernels.h5
wget -P tmp/weights/resnet/ http://ntu.h1fast.com/weights/resnet_50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

# mobilenet weights
wget -P tmp/weights/mobilenet/ http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224.h5
wget -P tmp/weights/mobilenet/ http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5
wget -P tmp/weights/mobilenet/ http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_224.h5
wget -P tmp/weights/mobilenet/ http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5
wget -P tmp/weights/mobilenet/ http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.3_224.h5
wget -P tmp/weights/mobilenet/ http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_224.h5

wget -P tmp/weights/mobilenet/ http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top.h5
wget -P tmp/weights/mobilenet/ http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5
wget -P tmp/weights/mobilenet/ http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_224_no_top.h5
wget -P tmp/weights/mobilenet/ http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
wget -P tmp/weights/mobilenet/ http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.3_224_no_top.h5
wget -P tmp/weights/mobilenet/ http://ntu.h1fast.com/weights/mobilenet/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_224_no_top.h5

# inception_resnet weights
wget -P tmp/weights/inception_resnet/ http://ntu.h1fast.com/weights/inception_resnet/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5
wget -P tmp/weights/inception_resnet/ http://ntu.h1fast.com/weights/inception_resnet/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5

# densenet weights
wget -P tmp/weights/densenet/ http://ntu.h1fast.com/weights/densenet/densenet169_weights_tf_dim_ordering_tf_kernels.h5
wget -P tmp/weights/densenet/ http://ntu.h1fast.com/weights/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5
wget -P tmp/weights/densenet/ http://ntu.h1fast.com/weights/densenet/densenet121_weights_tf_dim_ordering_tf_kernels.h5
wget -P tmp/weights/densenet/ http://ntu.h1fast.com/weights/densenet/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5
wget -P tmp/weights/densenet/ http://ntu.h1fast.com/weights/densenet/densenet201_weights_tf_dim_ordering_tf_kernels.h5
wget -P tmp/weights/densenet/ http://ntu.h1fast.com/weights/densenet/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5

# nasnet weights
wget -P tmp/weights/nasnet http://ntu.h1fast.com/weights/nasnet/NASNet-mobile.h5
wget -P tmp/weights/nasnet http://ntu.h1fast.com/weights/nasnet/NASNet-large-no-top.h5
wget -P tmp/weights/nasnet http://ntu.h1fast.com/weights/nasnet/NASNet-large.h5
wget -P tmp/weights/nasnet http://ntu.h1fast.com/weights/nasnet/NASNet-mobile-no-top.h5

```
