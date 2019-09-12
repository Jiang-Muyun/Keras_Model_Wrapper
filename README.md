# Segmentation and classification  warpper

This is a simple warpper for using deeplab and Mask-RCNN models in ROS easily. [Demo Video](https://www.youtube.com/watch?v=UnnYx1wMz68)

Models currently supported:
- Segmentation models
    + DeepLabv3 (Xception, MobileNetV2) [1]
    + Mask_RCNN (ResNet50) [2]
- Classification models [3]
    + VGG (VGG16 and VGG19)
    + ResNet50
    + Inception_V3
    + Xception
    + MobileNet_V2
    + Inception_ResNet_V2
    + Densenet (121, 169, 201)
    + NASNet (mobile and large)

## Installation

```bash
# create conda environment with python 3.6
conda create --name tf python=3.6
conda activate tf

# Install conda Jupyter notebooks supports.
conda install ipython nb_conda_kernels

# Install tensorflow, cuda and cudnn toolkit. We use cuda10 as example.
conda install -c anaconda tensorflow-gpu=1.14 cudatoolkit=10.0

# Install dependences
pip install cython
pip install Keras==2.2.4 tqdm Pillow scikit-image opencv-python h5py imgaug pycocotools requests
```

## Inference out of the box

```bash
# Clone this repo
git clone https://github.com/Jiang-Murray/Keras_Model_Wrapper.git
cd Keras_Model_Wrapper

# Activate environment for tensorflow applications
conda activate tf

# Run DeepLabv3 demo 
python deeplab_demo.py --mode images --model_name xception
python deeplab_demo.py --mode video --model_name mobilenetv2

# Run Mask_RCNN demo
python maskrcnn_demo.py --mode images
python maskrcnn_demo.py --mode video

# Run classification demo on imagenet
python classification.py
```

## Inference with ROS
The program can deal with both compressed and uncompressed image topics for inputs and outputs.

```bash
# Run Pascal_Voc trained DeepLabv3 Xception backbone on ROS topic and publish result to /deeplab/semantic/compressed
python ros/deeplab_node.py xception INPUT_TOPIC /deeplab/semantic/compressed

# Run Pascal_Voc trained DeepLabv3 Mobilenetv2 backbone on ROS topic and publish result to /deeplab/semantic/
python ros/deeplab_node.py mobilenetv2 INPUT_TOPIC /deeplab/semantic/compressed

# Run COCO trained Mask_RCNN on ROS topic
python ros/maskrcnn_node.py INPUT_TOPIC
```

## Working with rosbag

```bash
# Convert a rosbag to video
python ros/rosbag2video.py --topic INPUT_TOPIC

# Convert a rosbag to image sequences
python ros/rosbag2images.py --topic INPUT_TOPIC --interval 5
```

## Refenerce

1. https://github.com/bonlime/keras-deeplab-v3-plus
2. https://github.com/matterport/Mask_RCNN
3. https://github.com/keras-team/keras-applications