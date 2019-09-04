# Segmentation and classification  warpper

This is a simple warpper for using and evaluating deep learning models easily. [Demo Video](https://www.youtube.com/watch?v=UnnYx1wMz68)

Models currently supported:
- Segmentation models
    + DeepLabv3 (Xception, MobileNetV2)
    + Mask_RCNN ResNet50
- Classification models
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
The program can automatically identify compressed and uncompressed image topics for inputs and outputs.

```bash
# Run Pascal_Voc trained DeepLabv3 Xception backbone on ROS topic /camera/left/image_raw/compressed and publish a compressed color map image to /deeplab/semantic/compressed
python ros/deeplab_node.py xception /camera/left/image_raw/compressed /deeplab/semantic/compressed

# Run Pascal_Voc trained DeepLabv3 Mobilenetv2 backbone on ROS topic /camera/left/image_raw/compressed and publish a uncompressed color map image to /deeplab/semantic/
python ros/deeplab_node.py mobilenetv2 /camera/left/image_raw/compressed /deeplab/semantic/

# Run COCO trained Mask_RCNN on ROS topic /camera/left/image_raw/compressed
python ros/maskrcnn_node.py /camera/left/image_raw/compressed
```
