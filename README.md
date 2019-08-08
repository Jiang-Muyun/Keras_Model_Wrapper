# Segmentation and classification simple warpper

Model currently supported [Demo Video](https://www.youtube.com/watch?v=UnnYx1wMz68)

- Segmentation models
    + DeepLabv3 (Xception, MobileNetV2)
    + Mask_RCNN ResNet50
- Classification models
    + ResNet50
    + MobileNetV2
    + Xception
    + InceptionV3
    + Inception_ResnetV2
    + VGG16, VGG19
    + Densenet
    + NASNet

## Get Started

```bash
conda create --name tf python=3.6
conda activate tf

conda install ipython
conda install -c anaconda tensorflow-gpu cudatoolkit=10.0

pip install cython
pip install keras tqdm Pillow scikit-image opencv-python h5py imgaug pycocotools requests
```

The program will automatically download needed files. But the files can only be accessible inside NTU campus.

```bash
# Run DeepLabv3 demo on video
python deeplab_video.py

# Run Mask_RCNN demo on video
python mask_rcnn_video.py

# Run classification demo on imagenet
python imagenet_demo.py
```
