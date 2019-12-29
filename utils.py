import cv2
import os
import numpy as np
import itertools


# colour map
label_colours = [(0,0,255), (51,170,221),(0,255,255), (85,255,170),(170,255,85)]
# face, left arm, right arm, left_c, right_c

label_colours_top_bottom = [(255,85,0), (0,0,85),(0,119,221), (0,85,85),(85,51,0), (52,86,128),(0,128,0)]


def resize_and_pad(im, desired_size):
    # old_size is in (height, width) format
    old_size = im.shape[:2]
    ratio = float(desired_size) / max(old_size)

    # new_size should be in (width, height) format
    new_size = tuple([int(x * ratio) for x in old_size])
    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    return new_im, [top, bottom, left, right]


def get_bbox_roi(path):
    bbox_roi = []
    im = cv2.imread(os.path.join(path))
    h, w, c = im.shape
    total_mask = np.zeros(dtype=np.uint8, shape=(h, w))
    for i in label_colours_top_bottom:
        mask = cv2.inRange(im, *[i] * 2)
        total_mask = cv2.add(total_mask, mask)
    x, y, w, h = cv2.boundingRect(total_mask)
    bbox_roi.append((x, y, w, h))
    for i in label_colours:
        mask = cv2.inRange(im, i, i)
        x, y, w, h = cv2.boundingRect(mask)
        bbox_roi.append((x, y, w, h))

    return im, bbox_roi


def get_body_parts(im, bboxes, data_shape):
    lst_parts = []
    for x, y, w, h in bboxes:
        if x == y == w == h == 0:
            lst_parts.append(np.zeros(dtype=np.uint8, shape=(data_shape, data_shape, 3)) / 255.)
        else:
            im_crop = im[y: y + h, x: x + w]
            im_crop = im_crop / 255.
            im_crop, _ = resize_and_pad(im_crop, data_shape)
            lst_parts.append(im_crop)
    return lst_parts


def shuffle(df, n_split=5):
    df = df.copy()
    df = df.sample(frac=1).reset_index(drop=True)
    df['id'] = get_idx(len(df), n_split)
    return df


def get_idx(len_data, n_split=10):
    loop = len_data // n_split

    lst_idx = []
    for i in range(n_split):
        lst_idx += [*itertools.repeat(i, loop)]
    lst_idx = [*lst_idx, *itertools.repeat(n_split - 1, len_data % n_split)]
    return lst_idx


def filter_data_by_id(dataFrame, id_value):
    return dataFrame[dataFrame['id'] == id_value]


