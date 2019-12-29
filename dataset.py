import os
import cv2
import pandas as pd
import numpy as np

from config import Config as cfg
from utils import resize_and_pad

lst_name = ['sketch_1','sketch_2','origin_1','origin_2']


def convert_data_to_pd():
    rows = []
    columns = [lst_name[0], lst_name[1], lst_name[2], lst_name[3], "id"]

    dir_path = os.listdir(cfg.path_data)
    for i in dir_path:
        sub_dir = os.listdir(os.path.join(cfg.path_data,i))
        lst_items = [item for item in sub_dir]
        rows.append([os.path.join(cfg.path_data,lst_items[0]),
                     os.path.join(cfg.path_data,lst_items[1]),
                     os.path.join(cfg.path_data,lst_items[2]),
                     os.path.join(cfg.path_data,lst_items[3]),
                    "0"])

    df = pd.DataFrame(data=rows,columns=columns)
    df.to_csv(cfg.train_cvs, index=False)


def pre_processing(batchs):
    lst_sketch01 = []
    lst_sketch02 = []
    lst_origin01 = []
    lst_origin02 = []
    for index, row in batchs.iterrows():
        try:
            # get sketch 1
            item1 = cv2.imread(row[lst_name[0]])
            img_sketch01, _ = resize_and_pad(item1, cfg.data_shape)
            lst_sketch01.append(img_sketch01 / 255.)

            # get sketch 2
            item2 = cv2.imread(row[lst_name[1]])
            img_sketch02, _ = resize_and_pad(item2, cfg.data_shape)
            lst_sketch02.append(img_sketch02 / 255.)

            # get origin image 1 - label 1
            item3 = cv2.imread(row[lst_name[2]])
            img_origin01, _ = resize_and_pad(item3, cfg.data_shape)
            lst_origin01.append( (img_origin01 / 127.5) - 1 )

            # get origin image 1 - label 2
            item4 = cv2.imread(row[lst_name[3]])
            img_origin02, _ = resize_and_pad(item4, cfg.data_shape)
            lst_origin02.append((img_origin02 / 127.5) - 1)
        except Exception as e:
            print(e)
    return np.array(lst_sketch01), np.array(lst_sketch02), np.array(lst_origin01), np.array(lst_origin02)


if __name__ == '__main__':
    convert_data_to_pd()