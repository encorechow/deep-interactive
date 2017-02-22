import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

caffe_path = osp.join('caffe-di', 'python')
add_path(caffe_path)
