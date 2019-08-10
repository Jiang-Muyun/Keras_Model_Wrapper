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

## Get Started

```bash
# create conda environment with python 3.6
conda create --name tf python=3.6
conda activate tf

# Install conda Jupyter notebooks supports.
conda install ipython nb_conda_kernels

# Install tensorflow, cuda and cudnn toolkit. We use cuda10 as example.
conda install -c anaconda tensorflow-gpu cudatoolkit=10.0

# Install dependences
pip install cython
pip install keras tqdm Pillow scikit-image opencv-python h5py imgaug pycocotools requests
```

Time for some demo.

```bash
git clone https://github.com/Jiang-Murray/Keras_Model_Wrapper.git
cd Keras_Model_Wrapper
conda activate tf

# Run DeepLabv3 demo on video
python deeplab_video.py

# Run Mask_RCNN demo on video
python mask_rcnn_video.py

# Run classification demo on imagenet
python imagenet_demo.py
```
