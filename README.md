# Segmentation and classification simple warpper

The code is tested on python 3.6, tensorflow 1.14.0, Keras 2.2.4, Ubuntu 18.04. [Demo Video](https://www.youtube.com/watch?v=UnnYx1wMz68)

- Segmentation models supported
    + DeepLabv3 Xception
    + DeepLabv3 MobileNetV2 
    + Mask_RCNN ResNet50
- Classification models supported
    + Xception
    + Inception_v3
    + ResNet50
    + MobileNetV2
    + VGG16, VGG19
    + Inception_Resnet
    + Densenet
    + NASNet

## Get Started

Start with conda
```bash
conda create --name tf python=3.6
conda activate tf

conda install ipython
conda install -c anaconda tensorflow-gpu cudatoolkit=10.0

pip install cython
pip install keras tqdm Pillow scikit-image opencv-python h5py imgaug pycocotools requests
```

The program will automatically download files needed. But the files can only be accessible inside NTU campus.

```bash
# Run deeplab v3 demo on video
python deeplab_video.py

# Run mask_rcnn demo on video
python mask_rcnn_video.py

# Run classification demo on imagenet
python imagenet_demo.py
```
