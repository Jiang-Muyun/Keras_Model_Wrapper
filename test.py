from model_wrapper.utils import *
a = http_download(
    'tmp/weights/', 
    'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels.h5'
    ,skip_when_exists=False)
print(a)