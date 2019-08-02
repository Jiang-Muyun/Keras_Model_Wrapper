import json
import os
import sys
import cv2
import glob
import time
import random
import numpy as np
import matplotlib.pyplot as plt


def make_sure_path_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


make_sure_path_exist('tmp/')
make_sure_path_exist('tmp/img')

voc_samples = glob.glob('data/segmentation/*')
imagenet_samples = glob.glob('data/classification/*')

def sub_plot(fig, rows, cols, index, title, image):
    axis = fig.add_subplot(rows, cols, index)
    if title != None:
        axis.title.set_text(title)
    axis.axis('off')
    plt.imshow(image)


def auto_scale_to_display(batch):
    buf = []
    for x in batch:
        scaled = ((x - x.min())/(x.max()-x.min())*255.0).astype(np.uint8)
        buf.append(scaled)
    return np.array(buf, dtype=np.uint8)


def norm_01(x):
    return (x - x.min())/(x.max() - x.min())


def relu(x):
    return np.maximum(0, x)


def l2norm(x):
    return np.linalg.norm(x, ord=2)


def inf_norm(x):
    return np.linalg.norm(x, ord=np.inf_norm)


def np_clip_by_l2norm(x, clip_norm):
    return x * clip_norm / np.linalg.norm(x, ord=2)


def np_clip_by_infnorm(x, clip_norm):
    return x * clip_norm / np.linalg.norm(x, ord=np.inf)


def print_mat(x):
    print(x.shape, x.dtype, x.min(), x.max())


def multiple_randomint(min, max, count=1):
    assert count >= 1 and max > min
    buf = []
    for i in range(0, count):
        buf.append(random.randint(min, max))
    return buf


def vstack_images(fn_list,out_fn):
    buf = [cv2.imread(fn) for fn in fn_list]
    out = np.vstack(buf)
    cv2.imwrite(out_fn, out)


class Tick():
    def __init__(self, name='', silent=False):
        self.name = name
        self.silent = silent

    def __enter__(self):
        self.t_start = time.time()
        if not self.silent:
            print('> %s ... ' % (self.name), end='')
            sys.stdout.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t_end = time.time()
        self.delta = self.t_end-self.t_start
        self.fps = 1/self.delta

        if not self.silent:
            print('[%.0f ms]' % (self.delta * 1000))
            sys.stdout.flush()


class Tock():
    def __init__(self, name=None, report_time=True):
        self.name = '' if name == None else name+': '
        self.report_time = report_time

    def __enter__(self):
        self.t_start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t_end = time.time()
        self.delta = self.t_end-self.t_start
        self.fps = 1/self.delta
        if self.report_time:
            print('(%s%.0fms) ' % (self.name, self.delta * 1000), end='')
        else:
            print('.', end='')
        sys.stdout.flush()
