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

moduleBase = os.path.abspath(os.path.join(os.path.realpath(__file__), '../'))
if not moduleBase in sys.path:
    sys.path.append(moduleBase)

urls = json.load(open(os.path.join(moduleBase, 'data/urls.json')))

def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return '{:.1f} {}{}'.format(num, unit, suffix)
        num /= 1024.0
    return '{:.1f} {}{}'.format(num, 'Yi', suffix)

    
def auto_download(folder,tag):
    os.makedirs(folder,exist_ok = True)
    fn = urls[tag].split('/')[-1]
    local_filename = os.path.join(folder,fn)
    if os.path.exists(local_filename):
        return local_filename

    url = urls[tag]

    print('==> Downloading weights from Github')
    if not http_download(local_filename,url):
        if os.path.exists(local_filename):
            os.remove(local_filename)
        raise IOError('Unable to download pretrained weights from ' + urls[tag])
    return local_filename


def http_download(local_filename,url):
    bytes_downloaded = 0
    try:
        r = requests.get(url, stream=True, timeout=5)
        r.raise_for_status()
        t_start = time.time()
        with tqdm(total = int(r.headers['Content-Length'])) as pbar:
            with open(local_filename, 'wb') as fp:
                for chunk in r.iter_content(chunk_size=102400): 
                    pbar.update(len(chunk))
                    bytes_downloaded += len(chunk)
                    speed = int(bytes_downloaded /(time.time() - t_start))
                    status = '  %s (%s/s)'%(sizeof_fmt(bytes_downloaded), sizeof_fmt(speed))
                    pbar.set_description(status)
                    if chunk:
                        fp.write(chunk)
        return True
    except Exception as err:
        print(err)
        return False

def new_session():
    import tensorflow as tf
    import keras
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    return sess

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

class VOC_Utill():
    def __init__(self):
        moduleBase = os.path.abspath(os.path.join(os.path.realpath(__file__), '../'))
        tmp = json.load(open(os.path.join(moduleBase,'data/pascal_voc.json'), 'r'))
        self.num_classes = tmp['num_classes']
        self.labels = tmp['labels_short']
        self.labels_index = tmp['labels_index']
        self.colors = tmp['colors']
        self.colormap = np.array(self.colors, dtype=np.uint8)

    def get_label_colormap(self,label):
        assert label.dtype in [np.uint8, np.uint16, np.uint32], label.dtype
        assert label.max() <= 20 and label.min() >= 0, 'invalid range'
        return self.colormap[label]

    def show_legend(self):
        fig = plt.figure(figsize=(12, 3), dpi=80, facecolor='w', edgecolor='k')
        for index, (label, color) in enumerate(zip(self.labels, self.colors)):
            patch = np.full((32, 32, 3), color, dtype=np.uint8)
            axis = fig.add_subplot(2, 11, index+1)
            axis.title.set_text(label)
            axis.axis('off')
            plt.imshow(patch)
        plt.show(block=False)

    def semantic_report(self,semantic, limit=3):
        assert semantic.dtype == np.uint8, semantic.dtype
        assert semantic.shape == (512, 512), semantic.shape 
        assert semantic.min() >= 0 and semantic.max() <= 20 , 'invalid range'
        report = ''
        unique, counts = np.unique(semantic, return_counts=True)
        sort_index = np.argsort(np.array(counts)).tolist()
        sort_index.reverse()
        report_count = 0
        for index in sort_index:
            class_id = unique[index]
            count = counts[index]
            percent = count / (512.0*512.0) * 100.0
            if class_id == 0:
                continue
            if percent > 0.1:
                report += '%s:%.1f%% ' % (self.labels[class_id], percent)
                report_count += 1
                if report_count == limit:
                    break
        return report

    def semantic_classwise_distribution(self,batch):
        assert batch.dtype == np.uint8, batch.dtype
        assert batch.shape[1:] == (512, 512), batch.shape
        assert batch.min() >= 0 and batch.max() <= 20
        buf = []
        for i in range(0, batch.shape[0]):
            semantic = batch[i]
            distribution = np.zeros((21), dtype=np.uint8)
            unique, counts = np.unique(semantic, return_counts=True)
            for index, semantic_class in enumerate(unique):
                distribution[semantic_class] = counts[index] / (512.0*512.0) * 100
            buf.append(distribution)
        return np.array(buf, dtype=np.uint8)
    
voc = VOC_Utill()
    
def get_target(d_class=8):
    img = np.zeros((512, 512), dtype=np.uint8)
    #cv2.putText(img,'ICRA',(30,250), cv2.FONT_HERSHEY_COMPLEX, 6,(d_class*10),16,cv2.LINE_8)
    #cv2.putText(img,'2020',(30,400), cv2.FONT_HERSHEY_COMPLEX, 6,((d_class+1)*10),16,cv2.LINE_8)

    # cv2.rectangle(img,(150,150),(380,380),(d_class*10),-1)
    # cv2.circle(img,(256,256),150,(d_class*10),-1)

    # cv2.circle(img,(256,256),200,((d_class-1)*10),30)
    # cv2.circle(img,(256,256),120,(d_class*10),30)
    # cv2.circle(img,(256,256),50,((d_class+1)*10),-1)

    cv2.putText(img, 'A', (120, 410), cv2.FONT_HERSHEY_SIMPLEX,15, (d_class*10), 45, cv2.LINE_8)
    #_,img = cv2.threshold(img,127,d_class,cv2.THRESH_BINARY)
    #img = np.full((512,512),d_class,dtype=np.uint8)
    target = np.around(img/10).astype(np.uint8)
    return target
    

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
