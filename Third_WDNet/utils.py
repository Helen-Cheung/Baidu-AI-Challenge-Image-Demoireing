import os
import cv2
import paddle
import mmcv
import numpy as np
from math import ceil

def load_pretrained_model(model, pretrained_model):
    if pretrained_model is not None:
        print('Loading pretrained model from {}'.format(pretrained_model))

        if os.path.exists(pretrained_model):
            para_state_dict = paddle.load(pretrained_model)

            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    print("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(
                        model_state_dict[k].shape):
                    print(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape,
                                model_state_dict[k].shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            print("There are {}/{} variables loaded into {}.".format(
                num_params_loaded, len(model_state_dict),
                model.__class__.__name__))

        else:
            raise ValueError(
                'The pretrained model directory is not Found: {}'.format(
                    pretrained_model))
    else:
        print(
            'No pretrained model to load, {} will be trained from scratch.'.
            format(model.__class__.__name__))


def calculate_psnr(img1, img2, crop_border=0, input_order='HWC', convert_to=None):
    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    if isinstance(convert_to, str) and convert_to.lower() == 'y':
        img1 = mmcv.bgr2ycbcr(img1 / 255., y_only=True) * 255.
        img2 = mmcv.bgr2ycbcr(img2 / 255., y_only=True) * 255.
    elif convert_to is not None:
        raise ValueError('Wrong color model. Supported values are '
                         '"Y" and None.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    mse_value = np.mean((img1 - img2)**2)
    if mse_value == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse_value))


def sharp(images, method='lap_5'):
    if method == 'lap_9':
        lap = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    elif method == 'lap_5':
        lap = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    else:
        raise TypeError('Unknown')
    dst = cv2.filter2D(images, -1, kernel=lap)
    return dst


def chop_forward(model, inp, shave=8, min_size=160000):
    _, _, h, w = inp.shape
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    
    mod_size = 4
    if h_size%mod_size:
        h_size = ceil(h_size/mod_size)*mod_size  # The ceil() function returns the uploaded integer of a number
    if w_size%mod_size:
        w_size = ceil(w_size/mod_size)*mod_size
        
    inputlist = [
        inp[:, :, 0:h_size, 0:w_size],
        inp[:, :, 0:h_size, (w - w_size):w],
        inp[:, :, (h - h_size):h, 0:w_size],
        inp[:, :, (h - h_size):h,  (w - w_size):w] 
    ]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(4):
            with paddle.no_grad():
                input_batch = inputlist[i] 
                output_batch = model(input_batch)
            outputlist.append(output_batch) 
    else:
        outputlist = [
            chop_forward(model, patch) \
            for patch in inputlist]

    scale=1
    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with paddle.no_grad(): 
        output_ht = paddle.zeros_like(inp)

    output_ht[:, :, 0:h_half, 0:w_half] = outputlist[0][:, :, 0:h_half, 0:w_half]
    output_ht[:, :, 0:h_half, w_half:w] = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output_ht[:, :, h_half:h, 0:w_half] = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output_ht[:, :, h_half:h, w_half:w] = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output_ht


def average_inference(path1, path2, output = "./output/pre"):
    img_list1 = os.listdir(path1)
    img_list1 = [w for w in img_list1 if w.endswith('jpg')]
    img_list1.sort()
    img_list1 = [os.path.join(path1, w) for w in img_list1]

    img_list2 = os.listdir(path2)
    img_list2 = [w for w in img_list2 if w.endswith('jpg')]
    img_list2.sort()
    img_list2 = [os.path.join(path2, w) for w in img_list2]

    if not os.path.exists(output):
        os.mkdir(output)

    for i in range(len(img_list1)):
        print("Index: {}".format(i + 1))
        im1 = cv2.imread(img_list1[i])
        
        im2 = cv2.imread(img_list2[i])
        im2 = cv2.flip(im2, 90)

        im1 = im1.astype(np.float32)
        im2 = im2.astype(np.float32)

        im = (im1 + im2) / 2
        im = im.astype(np.uint8)

        im_name = os.path.join(output, os.path.basename(img_list1[i]))
        cv2.imwrite(im_name, im)