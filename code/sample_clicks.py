import os.path as osp
import os
from cfg.config import cfg
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdb
import random as ran
import argparse
from tifffile import imsave

def load_image_path(img_dir, img_ext):
    '''Load training image text file as list
    @img_dir: image directory name.
    @img_ext: image extension type.
    output: list of image file path
    '''
    images = []
    file_names = []
    if not img_dir:
        print('No image directory provided.')
        return images

    txt_path = osp.join(cfg.DATA_DIR, 'PASCAL','train.txt')
    with open(txt_path, 'r') as fp:
        for line in fp:
            # truncate /n character
            file_names.append(line[:-1])
            file_name = line[:-1] + img_ext
            images.append(osp.join(cfg.DATA_DIR, 'PASCAL', img_dir, file_name))

    return images, file_names


def euclidean_dis(p1, p2):
    return np.linalg.norm(p1-p2)


def sample_cands(candidates, is_fixed, num, criteria, *args):
    res = []

    if not is_fixed:
        num = ran.choice(range(num))

    # to prevent deadlock when there are no enough points to satisfy the criteria
    cnt = 0

    # Collecting num points
    while len(res) != num and cnt < 500:
        cnt += 1
        cand = ran.choice(candidates)

        # If satisfy criteria, add to result
        if criteria(res, cand, *args):
            res.append(cand)


    return res


def dis_criteria(points, point, least_dist):
    if len(points) == 0:
        return True

    for p in points:
        dis = euclidean_dis(point, p)

        # exit if the distance constraint is not satisfied
        if dis < least_dist:
            return False
    return True



def _compute_step(ys, xs):

        min_x, min_y, max_x, max_y = np.min(xs), np.min(ys), np.max(xs), np.max(ys)

        h = max_y - min_y
        w = max_x - min_x

        min_dist = min(h, w)

        d_step = (min_dist - cfg.D_MARGIN) // cfg.RATIO_FACTOR

        return d_step


def _erosion(valid_area):
    # Fixed 3 by 3 erosion
    erode_dim = (2 * cfg.D_MARGIN + 1, 2 * cfg.D_MARGIN + 1)
    erode_kernel = np.ones(erode_dim)
    erode = cv2.erode(valid_area, erode_kernel, iterations=1)
    return erode

def _dilation_and_remove(valid_area):
    dilation_dim = (2 * cfg.D + 1, 2 * cfg.D + 1)
    dilation_kernel = np.ones(dilation_dim)
    dilation = cv2.dilate(valid_area, dilation_kernel, iterations=1)
    dilation -= valid_area
    return dilation

def _dilation_and_remove_and_erosion(valid_area):
    dilation_dim = (2 * cfg.D + 1, 2 * cfg.D + 1)
    dilation_kernel = np.ones(dilation_dim)
    dilation = cv2.dilate(valid_area, dilation_kernel, iterations=1)
    dilation -= valid_area

    erode_dim = (2 * cfg.NEG3_MARGIN + 1, 2 * cfg.NEG3_MARGIN + 1)
    erode_kernel = np.ones(erode_dim)
    erode = cv2.erode(dilation, erode_kernel, iterations=1)


    return erode


def _filter_samples(path, is_fixed, morphology, num=cfg.N_POS):
    '''Sampling points from the pixels that get from morphology function for each object
    based on following rules:
        1. Any two pixels are at least d_step pixels away from each other;
        2. Any pixels is at least d_margin pixels away from object boundaries;

    @path: image path
    @morphology: function to get desired sample area
    output: a dictionary of positive points for each object per entry
    '''
    locs = {}
    if not path:
        return locs

    img_arr = cv2.imread(path, 0)

    obj_lst = np.unique(img_arr[img_arr != 0])


    d_step = 1

    obj_mappings = {}
    for obj in obj_lst:
        # numpy logic operation for masking
        obj_mappings[obj] = img_arr * (img_arr == obj)
        locs[obj] = []


    for obj, mapping in obj_mappings.items():


        # Get desired sample area
        morph = morphology(mapping)

        # Prevent the segmentation is too thin to erode.
        if np.sum(morph) == 0:
            morph = np.array(mapping, dtype=np.uint8)

        ys, xs = np.where(morph == obj)
        d_step = _compute_step(ys, xs)


        # Find indexes of where the element equal to obj.
        candidates = np.array(list(zip(*np.where(morph== obj))))


        # Sample points from candidates
        samples = sample_cands(candidates, is_fixed, num, dis_criteria, d_step)
        while len(samples) == 0 and num == cfg.N_POS:
            samples = sample_cands(candidates, is_fixed, num, dis_criteria, d_step)

        locs[obj].extend(samples)

        #for s in samples:
        #    morph[s[0], s[1]] = 200
        #print(samples)
        #plt.figure()
        #plt.subplot(221)
        #plt.imshow(morph)
        #plt.subplot(222)
        #plt.imshow(mapping)
        #plt.show()

    return locs


def pos_strategy(path):
    '''Positive points sampling strategy.
    @path: image path
    @output: a dictionary of sampled negative points
    '''
    return _filter_samples(path, False, _erosion)


def neg_strategy1(path, num_negs=10):
    '''Negative points sampling strategy 1.
    @num_negs: number of negative points needed to be sampled
    @path: image path
    output: a dictionary of sampled negative points
    '''
    return _filter_samples(path, False, _dilation_and_remove, num_negs)


def neg_strategy2(path, num_negs=5):
    '''Negative points sampling strategy 2.
    @num_negs: number of negative points needed to be sampled
    @path: idef f(x):
    return {
        'a': 1,
        'b': 2,
    }[x]mage path
    output: a dictionary of sampled negative points
    '''
    locs = {}
    if not path:
        return locs

    img_arr = cv2.imread(path, 0)

    obj_lst = np.unique(img_arr[img_arr != 0])


    obj_mappings = {}
    for obj in obj_lst:
        obj_mappings[obj] = img_arr * (img_arr == obj)
        locs[obj] = []

    for obj, mapping in obj_mappings.items():


        for other_obj, other_mapping in obj_mappings.items():
            if obj != other_obj:

                morph = _erosion(other_mapping)

                if np.sum(morph) == 0:
                    morph = np.array(other_mapping, dtype=np.uint8)

                ys, xs = np.where(morph == other_obj)

                d_step = _compute_step(ys, xs)

                candidates = np.array(list(zip(*np.where(morph == other_obj))))

                samples = sample_cands(candidates, False, num_negs, dis_criteria, d_step)

                locs[obj].extend(samples)

                #for s in samples:
                #    morph[s[0], s[1]] = 200
                #print(samples)
                #plt.figure()
                #plt.subplot(221)
                #plt.imshow(morph)
                #plt.subplot(222)
                #plt.imshow(mapping)
                #plt.show()

    return locs

def neg_strategy3(path, num_negs=10):
    '''Negative points sampling strategy 3.
    @num_negs: number of negative points needed to be sampled
    @path: image path
    output: a dictionary of sampled negative points
    '''
    return _filter_samples(path, True, _dilation_and_remove_and_erosion, num_negs)


def cat_channels(image, gt):
    '''Constructing positive mapping channel and negative mapping channels.
    input:
    output:
    '''

    res = {}
    res['data'] = []
    res['labels'] = []
    if not gt or not image:
        return None

    gt_arr = cv2.imread(gt, 0)

    h, w = gt_arr.shape

    img_arr = cv2.imread(image)

    #print(img_arr.shape)

    for n in range(cfg.N_PAIRS):

        pos_samples = pos_strategy(gt)


        neg_samples = [neg_strategy1, neg_strategy2, neg_strategy3]

        idx = n // (cfg.N_PAIRS)

        neg_samples = neg_samples[idx](gt)


        pos_energies = construct_channels(pos_samples, h, w)

        neg_energies = construct_channels(neg_samples, h, w)


        for obj in pos_samples:
            pairs = np.concatenate((img_arr, pos_energies[obj], neg_energies[obj]), axis=2)
            label = gt_arr * (gt_arr == obj)
            res['data'].append(pairs)
            res['labels'].append(label)

    return res


def construct_channels(samples, h, w):

    energies = {}
    for obj, lst in samples.items():
        energy = np.zeros((h, w))
        energy.fill(255)

        for p in lst:
            y_vec = np.arange(h)

            x_vec = np.arange(w)

            y_vec = y_vec[:, np.newaxis]
            x_vec = x_vec[:, np.newaxis]

            y_vec -= p[0]
            x_vec -= p[1]

            intermedia = np.power(y_vec,2) + np.power(x_vec.T, 2)
            intermedia = np.sqrt(intermedia)

            intermedia[intermedia > 255] = 255

            energy = np.array((intermedia, energy), dtype=np.uint8)

            energy = energy.min(axis=0)


        energy = energy[:, :, np.newaxis]

        energies[obj] = energy

    return energies

def argparser():
    ''' Parse all arguments provided from the command line tool(CLT)
    '''
    parser = argparse.ArgumentParser(description='Convert images to five channels mapping.')
    parser.add_argument('--save-dir', type=str, default='converted', help="Directory to store converted tiff files")
    return parser.parse_args()


def create_dir(dir_name):
    if not osp.exists(dir_name):
        os.makedirs(dir_name)

def main():

    args = argparser()

    gts, filenames = load_image_path(cfg.GT_DIR, cfg.GT_EXT)
    images, _ = load_image_path(cfg.IMG_DIR, cfg.IMG_EXT)

    output_path = osp.join(cfg.DATA_DIR, cfg.BENCHMARK_DIR, args.save_dir)

    create_dir(output_path)

    labels_dir = osp.join(output_path, 'labels')

    create_dir(labels_dir)

    data_dir = osp.join(output_path, 'data')

    create_dir(data_dir)

    train_txt = open(osp.join(output_path, 'train.txt'), 'w+')
    train_txt.truncate()


    for idx, (image, gt, fn) in enumerate(zip(images, gts, filenames)):

        pairs = cat_channels(image, gt)


        data = np.array(pairs['data'], dtype=np.uint8)
        labels = np.array(pairs['labels'], dtype=np.uint8)

        for i in range(data.shape[0]):
            sample = data[i,...]
            label = labels[i,...]

            sample = sample.transpose((2, 0, 1))

            new_name = fn + '-' + str(i) + '.tif'
            new_label_name = fn + '-' + str(i) + '.tif'

            file_path = osp.join(data_dir, new_name)
            label_path = osp.join(labels_dir, new_label_name)

            train_txt.write(file_path + ' ' + label_path + '\n')

            imsave(file_path, sample, compress=6)
            imsave(label_path, label, compress=6)

        print('image {} done!, counter={}'.format(fn, idx))

    train_txt.close()

if __name__ == '__main__':
    main()


