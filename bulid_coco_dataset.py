import itertools

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import json


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


# From https://newbedev.com/encode-numpy-array-using-uncompressed-rle-for-coco-dataset
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def polygonFromMask(maskedArr):
    # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    valid_poly = 0
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
    return segmentation


def create_coco_format_json(data_frame, classes, filepaths):
    images = []
    annotations = []
    categories = []
    count = 0

    # Additing categories
    for idx, class_ in enumerate(classes):
        categories.append(
            {
                "id": idx,
                "name": class_
            }
        )
    i = 0
    for filepath in tqdm(filepaths):
        i += 1
        image_id = i
        file_id = ('_'.join((filepath.split("\\")[-3] + "_" + filepath.split("\\")[-1]).split("_")[:-4]))
        height_slice = int(filepath.split("\\")[-1].split("_")[3])
        width_slice = int(filepath.split("\\")[-1].split("_")[2])
        ids = data_frame.index[data_frame['id'] == file_id].tolist()
        if len(ids) > 0:
            # Adding images which has annotations
            images.append(
                {
                    "id": image_id,
                    "file_id": file_id,  # add id of the dataframe
                    "width": width_slice,
                    "height": height_slice,
                    "file_name": filepath
                }
            )
            for segm in ['rle_large', 'rle_small', 'rle_stomach']:
                for idx in ids:
                    if isinstance(data_frame.iloc[idx][segm], float):
                        continue
                    mk = rle_decode(data_frame.iloc[idx][segm], (height_slice, width_slice))
                    ys, xs = np.where(mk)
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    contours, hierarchy = cv2.findContours(mk, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                    for id_, contour in enumerate(contours):
                        mask_image = np.zeros((mk.shape[0], mk.shape[1], 3), np.uint8)
                        cv2.drawContours(mask_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
                        mask_image_bool = np.array(mask_image, dtype=bool).astype(np.uint8)
                        ys, xs = np.where(mask_image_bool)
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)
                        enc = polygonFromMask(mask_image_bool)
                        # enc = binary_mask_to_rle(mask_image_bool)
                        seg = {
                            'segmentation': enc,
                            'bbox': [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)],
                            'area': int(np.sum(mask_image_bool)),
                            'image_id': image_id,
                            'category_id': classes.index(segm),
                            'iscrowd': 0,
                            'id': count
                        }
                        annotations.append(seg)
                        count += 1
                    # creating the dataset
    dataset_coco_format = {
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    return dataset_coco_format


if __name__ == '__main__':
    train_df = pd.read_csv('./train_df.csv')
    test_df = pd.read_csv('./test_df.csv')

    classes = ['rle_large', 'rle_small', 'rle_stomach']
    train_json = create_coco_format_json(train_df, classes, train_df['img_path'].tolist())
    train_json = json.dumps(train_json)
    f1 = open('train_json.json', 'w')
    f1.write(train_json)
    f1.close()

    test_json = create_coco_format_json(test_df, classes, test_df['img_path'].tolist())
    test_json = json.dumps(test_json)
    f2 = open('test_json.json', 'w')
    f2.write(test_json)
    f2.close()
