from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import pandas as pd
import itertools
import numpy as np


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


config_file = 'configs2/uw_madison_gi_tract/胃肠道图像分割.py'
checkpoint_file = 'work_dirs/胃肠道图像分割/latest.pth'
model = init_detector(config_file, checkpoint_file)

sub_df = pd.read_csv("sample_submission.csv")
train_csv = pd.read_csv("train.csv")

labels = ['large_bowel', 'small_bowel', 'stomach']

for i in range(len(train_csv)):
    id = train_csv.loc[i]['id']
    img = train_csv.loc[i]['path']
    label = train_csv.loc[i]['class']
    seg = train_csv.loc[i]['segmentation']
    if isinstance(seg, float):
        index = labels.index(label)
        result = inference_detector(model, img)
        try:
            mask = result[1][index][0]
        except:
            continue
        rle_mask = binary_mask_to_rle(mask)['counts']
        sub_df.loc[i] = [id, label, rle_mask]
    if i % 1000 == 0:
        print(i)
sub_df.to_csv("sample_submission.csv")
