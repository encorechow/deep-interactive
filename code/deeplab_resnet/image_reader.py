import os
import os.path as osp

import numpy as np
import tensorflow as tf

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks

