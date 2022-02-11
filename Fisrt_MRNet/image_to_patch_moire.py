"""
This code is used to extract image patches from the training and validation
sets as described in the paper. For the training set patches, we discard 30%
of the patches that have the lowest sharpness energy. Recall that we don't
extract patches for test images because we process full image at test time.

Copyright (c) 2020-present, Abdullah Abuolaim
This source code is licensed under the license found in the LICENSE file in
the root directory of this source tree.

Note: this code is the implementation of the "Defocus Deblurring Using Dual-
Pixel Data" paper accepted to ECCV 2020. Link to GitHub repository:
https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel

Email: abuolaim@eecs.yorku.ca
"""

import numpy as np
import os
import cv2
import errno
from copy import deepcopy


def check_create_directory(path_to_check):
    if not os.path.exists(path_to_check):
        try:
            os.makedirs(path_to_check)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def shapness_measure(img_temp, kernel_size):
    conv_x = cv2.Sobel(img_temp, cv2.CV_64F, 1, 0, ksize=kernel_size)
    conv_y = cv2.Sobel(img_temp, cv2.CV_64F, 0, 1, ksize=kernel_size)
    temp_arr_x = deepcopy(conv_x * conv_x)
    temp_arr_y = deepcopy(conv_y * conv_y)
    temp_sum_x_y = temp_arr_x + temp_arr_y
    temp_sum_x_y = np.sqrt(temp_sum_x_y)
    return np.sum(temp_sum_x_y)


def filter_patch_sharpness(patches_src_c_temp, patches_trg_c_temp):
    global patches_src_c, patches_trg_c, patches_src_l, patches_src_r
    fitnessVal_3 = []
    fitnessVal_7 = []
    fitnessVal_11 = []
    fitnessVal_15 = []
    num_of_img_patches = len(patches_trg_c_temp)
    for i in range(num_of_img_patches):
        fitnessVal_3.append(shapness_measure(cv2.cvtColor(patches_trg_c_temp[i], cv2.COLOR_BGR2GRAY), 3))
        fitnessVal_7.append(shapness_measure(cv2.cvtColor(patches_trg_c_temp[i], cv2.COLOR_BGR2GRAY), 7))
        fitnessVal_11.append(shapness_measure(cv2.cvtColor(patches_trg_c_temp[i], cv2.COLOR_BGR2GRAY), 11))
        fitnessVal_15.append(shapness_measure(cv2.cvtColor(patches_trg_c_temp[i], cv2.COLOR_BGR2GRAY), 15))
    fitnessVal_3 = np.asarray(fitnessVal_3)
    fitnessVal_7 = np.asarray(fitnessVal_7)
    fitnessVal_11 = np.asarray(fitnessVal_11)
    fitnessVal_15 = np.asarray(fitnessVal_15)
    fitnessVal_3 = (fitnessVal_3 - np.min(fitnessVal_3)) / np.max((fitnessVal_3 - np.min(fitnessVal_3)))
    fitnessVal_7 = (fitnessVal_7 - np.min(fitnessVal_7)) / np.max((fitnessVal_7 - np.min(fitnessVal_7)))
    fitnessVal_11 = (fitnessVal_11 - np.min(fitnessVal_11)) / np.max((fitnessVal_11 - np.min(fitnessVal_11)))
    fitnessVal_15 = (fitnessVal_15 - np.min(fitnessVal_15)) / np.max((fitnessVal_15 - np.min(fitnessVal_15)))
    fitnessVal_all = fitnessVal_3 * fitnessVal_7 * fitnessVal_11 * fitnessVal_15

    to_remove_patches_number = int(to_remove_ratio * num_of_img_patches)

    for itr in range(to_remove_patches_number):
        minArrInd = np.argmin(fitnessVal_all)
        fitnessVal_all[minArrInd] = 2
    for itr in range(num_of_img_patches):
        if fitnessVal_all[itr] != 2:
            patches_src_c.append(patches_src_c_temp[itr])
            patches_trg_c.append(patches_trg_c_temp[itr])


def slice_stride(_img_src_c, _img_trg_c):
    global set_type, patch_size, stride, patches_src_c, patches_trg_c, patches_src_l, patches_src_r
    coordinates_list = []
    coordinates_list.append([0, 0, 0, 0])
    patches_src_c_temp, patches_trg_c_temp, patches_src_l_temp, patches_src_r_temp = [], [], [], []
    for r in range(0, _img_src_c.shape[0], stride[0]):
        for c in range(0, _img_src_c.shape[1], stride[1]):
            if (r + patch_size[0]) <= _img_src_c.shape[0] and (c + patch_size[1]) <= _img_src_c.shape[1]:
                patches_src_c_temp.append(_img_src_c[r:r + patch_size[0], c:c + patch_size[1]])
                patches_trg_c_temp.append(_img_trg_c[r:r + patch_size[0], c:c + patch_size[1]])
            elif (r + patch_size[0]) <= _img_src_c.shape[0] and not (
                    [r, r + patch_size[0], _img_src_c.shape[1] - patch_size[1],
                     _img_src_c.shape[1]] in coordinates_list):
                patches_src_c_temp.append(
                    _img_src_c[r:r + patch_size[0], _img_src_c.shape[1] - patch_size[1]:_img_src_c.shape[1]])
                patches_trg_c_temp.append(
                    _img_trg_c[r:r + patch_size[0], _img_trg_c.shape[1] - patch_size[1]:_img_trg_c.shape[1]])
                coordinates_list.append(
                    [r, r + patch_size[0], _img_src_c.shape[1] - patch_size[1], _img_src_c.shape[1]])
            elif (c + patch_size[1]) <= _img_src_c.shape[1] and not (
                    [_img_src_c.shape[0] - patch_size[0], _img_src_c.shape[0], c,
                     c + patch_size[1]] in coordinates_list):
                patches_src_c_temp.append(
                    _img_src_c[_img_src_c.shape[0] - patch_size[0]:_img_src_c.shape[0], c:c + patch_size[1]])
                patches_trg_c_temp.append(
                    _img_trg_c[_img_trg_c.shape[0] - patch_size[0]:_img_trg_c.shape[0], c:c + patch_size[1]])
                coordinates_list.append(
                    [_img_src_c.shape[0] - patch_size[0], _img_src_c.shape[0], c, c + patch_size[1]])
            elif not ([_img_src_c.shape[0] - patch_size[0], _img_src_c.shape[0], _img_src_c.shape[1] - patch_size[1],
                       _img_src_c.shape[1]] in coordinates_list):
                patches_src_c_temp.append(_img_src_c[_img_src_c.shape[0] - patch_size[0]:_img_src_c.shape[0],
                                          _img_src_c.shape[1] - patch_size[1]:_img_src_c.shape[1]])
                patches_trg_c_temp.append(_img_trg_c[_img_trg_c.shape[0] - patch_size[0]:_img_trg_c.shape[0],
                                          _img_trg_c.shape[1] - patch_size[1]:_img_trg_c.shape[1]])
                coordinates_list.append(
                    [_img_src_c.shape[0] - patch_size[0], _img_src_c.shape[0], _img_src_c.shape[1] - patch_size[1],
                     _img_src_c.shape[1]])

    if set_type == 'train':
        filter_patch_sharpness(patches_src_c_temp, patches_trg_c_temp)
        # patches_src_c, patches_trg_c = deepcopy(patches_src_c_temp), deepcopy(patches_trg_c_temp)
    else:
        patches_src_c, patches_trg_c = deepcopy(patches_src_c_temp), deepcopy(patches_trg_c_temp)


set_type_arr = ['train']
img_ex = '.jpg'
sub_folder = ['images/', 'gts/']
dataset = './moire_train_dataset/'

# color flag used to select the reading image mode in opencv. 0:graysca,
# 1:rgb 8bits, -1:read image as it including its bit depth
color_flag = -1

patch_size = [512, 512]

to_remove_ratio = 0.2  # discard 30% of the patches

for set_type in set_type_arr:
    print('Image to patch of ', set_type, 'set has started...')
    if set_type == 'train':
        # patch settings
        patch_overlap_ratio = 0.6
        stride = [int((1 - patch_overlap_ratio) * patch_size[0]), int((1 - patch_overlap_ratio) * patch_size[1])]
    else:
        # patch settings
        patch_overlap_ratio = 0.01
        stride = [int((1 - patch_overlap_ratio) * patch_size[0]), int((1 - patch_overlap_ratio) * patch_size[1])]

    # pathes to write extracted patches
    path_write_g = './moire_all_patch/' + '/gts/'
    path_write_h = './moire_all_patch/' + '/images/'

    # to check if directory exist, otherwise create one
    check_create_directory(path_write_g)
    check_create_directory(path_write_h)

    # load image filenames
    images_src = os.listdir(dataset + '/gts/')

    # set counter
    img_patch_count = 0

    data_ims_size = len(images_src)
    for i in range(data_ims_size):
        patches_src_c = []
        patches_trg_c = []

        img_trg_c = cv2.imread(dataset + '/gts/' + images_src[i], 1)
        img_src_c = cv2.imread(dataset + '/images/' + images_src[i], 1)

        print(dataset + '/gts/' + images_src[i])

        slice_stride(img_src_c, img_trg_c)
        for j in range(len(patches_src_c)):
            cv2.imwrite(path_write_h + str(img_patch_count).zfill(5) + img_ex, patches_src_c[j])
            cv2.imwrite(path_write_g + str(img_patch_count).zfill(5) + img_ex, patches_trg_c[j])

            img_patch_count += 1
            print(set_type + ': ', i, j, img_patch_count)
