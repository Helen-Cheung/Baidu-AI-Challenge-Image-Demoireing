import os
import pickle
import random
import numpy as np
import cv2
import scipy.ndimage
import glob
####################
# Files & IO
####################

###################### get image path list ######################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', 'npy']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def _get_paths_from_lmdb(dataroot):
    '''get image path list from lmdb meta info'''
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes


def get_image_paths(data_type, dataroot):
    '''get image path list
    support lmdb or image files'''
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths, sizes = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return sizes, paths

###################### read images ######################
def _read_img_lmdb(env, key, size):
    '''read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple'''
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img


def read_img(path, size=None):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def bgr2yuv_split(bgr):
    """
    Convert the bgr channel to yuv.
    :param bgr: bgr channel of the video frame.
    :return: y(float32) and  u and v channel.
    """
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)

    y = y.astype(np.float32)
    y = np.asarray(y)

    return y, u, v

def read_condition(path):
    cond = np.load(path, allow_pickle=True).astype(np.float32)
    return cond

def adjust_contrast(img, alpha):
    # input: RGB image, np.array, [0, 255]
    # output: RGB image, np.array, [0, 255]
    if img.shape[2] == 1:  # gray image
        img_mean = np.mean(img)
        img = alpha*img + (1-alpha)*img_mean
        return img
    elif img.shape[2] == 3: # RGB image
        img_mean_0 = np.mean(img[:,:,0])
        img_mean_1 = np.mean(img[:,:,1])
        img_mean_2 = np.mean(img[:,:,2])
        img[:,:,0] = np.clip(alpha*img[:,:,0] + (1-alpha)*img_mean_0, 0 ,1)
        img[:,:,1] = np.clip(alpha*img[:,:,1] + (1-alpha)*img_mean_1, 0 ,1)
        img[:,:,2] = np.clip(alpha*img[:,:,2] + (1-alpha)*img_mean_2, 0 ,1)
        return img


def adjust_brightness(img, alpha):
    # input: RGB image, np.array, [0, 255]
    # output: RGB image, np.array, [0, 255]
    if img.shape[2] == 1:  # gray image
        return img*alpha
    elif img.shape[2] == 3:  # RGB image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img[:,:,0] = np.clip(img[:,:,0]*alpha, 0, 1)  # Y*alpha
        # img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        return img


def color_saturation_v1(image_rgb, b):
    degenerate = image_rgb.mean(axis=2)
    image_rgb[:, :, 0] = np.clip(b * image_rgb[:, :, 0] + (1 - b) * degenerate, 0, 1)
    image_rgb[:, :, 1] = np.clip(b * image_rgb[:, :, 1] + (1 - b) * degenerate, 0, 1)
    image_rgb[:, :, 2] = np.clip(b * image_rgb[:, :, 2] + (1 - b) * degenerate, 0, 1)
    image_rgb = np.clip(image_rgb, 0, 1)
    return image_rgb


def tone_mapping(img, L=4, t=[3/8, 2/8, 1/8, 2/8]):
    assert len(t) == L
    assert np.sum(t) == 1

    sum = np.zeros(img.shape)
    for i in range(L):
        sum += np.clip(L * img - i, 0, 1) * t[i]
    img = sum

    return img


# def color_saturation_v2(img, alpha):
#     img = np.array(np.clip(img * 255, 0, 255), dtype=np.uint8)
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#     s = img_hsv[:, :, 1]
#     v = img_hsv[:, :, 2]
#     s = s / 255.0
#     v = v / 255.0
#     enhanced_s = s + (1 - s) * (0.5 - np.abs(0.5 - v)) * alpha
#     enhanced_s = np.array(np.clip(enhanced_s * 255, 0, 255), dtype=np.uint8)
#     enhanced_img_hsv = np.stack([img_hsv[:, :, 0], enhanced_s, img_hsv[:, :, 2]], 2)
#     img_rgb = cv2.cvtColor(enhanced_img_hsv, cv2.COLOR_HSV2RGB)
#     img_rgb = img_rgb / 255.0
#
#     return img_rgb




####################
# image processing
# process on numpy image
####################


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def augment_flow(img_list, flow_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:
            flow = flow[:, ::-1, :]
            flow[:, :, 0] *= -1
        if vflip:
            flow = flow[::-1, :, :]
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    rlt_img_list = [_augment(img) for img in img_list]
    rlt_flow_list = [_augment_flow(flow) for flow in flow_list]

    return rlt_img_list, rlt_flow_list


def channel_convert(in_c, tar_type, img_list):
    # conversion among BGR, gray and y
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == 'y':  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def filtering(img_gray, r, eps):
    img = np.copy(img_gray)
    H = 1/np.square(r)*np.ones([r,r])
    meanI = scipy.ndimage.correlate(img, H, mode='nearest')

    var = scipy.ndimage.correlate(img*img, H, mode='nearest') - meanI*meanI
    a = var/(var+eps)
    b = meanI-a*meanI

    meana = scipy.ndimage.correlate(a, H, mode='nearest')
    meanb = scipy.ndimage.correlate(b, H, mode='nearest')
    output = meana*img + meanb
    return output

def guided_filter(img_LR, r=5, eps=0.01):
    img = np.copy(img_LR)
    for i in range(3):
        img[:,:,i] = filtering(img[:,:,i], r, eps)
    return img
