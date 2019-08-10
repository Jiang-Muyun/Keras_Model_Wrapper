import json
import os
import sys
import cv2
import glob
import time
import random
import requests
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

moduleBase = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../'))
if not moduleBase in sys.path:
    sys.path.append(moduleBase)

urls = json.load(open(os.path.join(moduleBase, 'data/urls.json')))
urls_cache = json.load(open(os.path.join(moduleBase, 'data/urls_cache.json')))
voc_samples = glob.glob(os.path.join(moduleBase, 'data/segmentation/*'))
imagenet_samples = glob.glob(os.path.join(moduleBase, 'data/classification/*'))

def auto_download(folder,tag):
    os.makedirs(folder,exist_ok = True)
    fn = urls[tag].split('/')[-1]
    local_filename = os.path.join(folder,fn)
    if os.path.exists(local_filename):
        print('> Use cache',local_filename)
        return local_filename
    
    try:
        r = requests.get('https://ntu.h1fast.com/index.html', timeout=0.5)
        r.raise_for_status()
        url = urls_cache[tag]
        print('==> Downloading weights from cache server')
    except:
        url = urls[tag]
        print('==> Downloading weights from Github')

    if not http_download(local_filename,url):
        if os.path.exists(local_filename):
            os.remove(local_filename)
        raise Exception('Unable to download pretrained weights from ' + urls[tag])
    return local_filename

def http_download(local_filename,url):
    size_downloaded = 0
    try:
        r = requests.get(url, stream=True, timeout=5)
        r.raise_for_status()
        with tqdm(total = int(r.headers['Content-Length'])) as pbar:
            with open(local_filename, 'wb') as fp:
                for chunk in r.iter_content(chunk_size=102400): 
                    pbar.update(len(chunk))
                    size_downloaded += len(chunk)
                    pbar.set_description("> %.2f MB" % (size_downloaded/1024/1024))
                    if chunk:
                        fp.write(chunk)
        return True
    except Exception as err:
        print(err)
        return True
    except KeyboardInterrupt:
        return False


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
