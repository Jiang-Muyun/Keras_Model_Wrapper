import time
import numpy as np
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from utils import voc, sub_plot
import cv2

# model = models.segmentation.fcn_resnet101(pretrained=True).cuda().eval()
model = models.segmentation.deeplabv3_resnet101(pretrained=1).cuda().eval()

image = Image.open('data/voc/test1.jpg')
input_shape = (image.size[1], image.size[0])

img = np.array(image)
transform = T.Compose([T.Resize(224),
                       T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
img_batch = transform(image).cuda().unsqueeze(0)

with torch.no_grad():
    logits = model(img_batch)['out']
    inter = F.interpolate(logits, input_shape, mode='bilinear', align_corners=False)
    pred = inter.max(axis=1)[1]
    print(pred.cpu().numpy().shape, pred.cpu().numpy().dtype)
    disp = voc.get_label_colormap(pred.squeeze(0).cpu().numpy())

print(img.shape, disp.shape)
overlap = cv2.addWeighted(img, 0.5, disp, 0.5, 20)

fig = plt.figure(figsize=(12, 4), dpi=100, facecolor='w', edgecolor='k')
sub_plot(fig, 1, 3, 1, 'image', img)
sub_plot(fig, 1, 3, 2, voc.semantic_report(pred.squeeze(0).cpu().numpy().astype(np.uint8)), disp)
sub_plot(fig, 1, 3, 3, 'overlap', overlap)
plt.show(block=False)

plt.show()
