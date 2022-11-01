from sonic_ai.pipelines.init_pipeline import *
from mmdet.datasets.pipelines.compose import Compose


label_path = r"D:\Data\label.ini"
dataset_path_list = ["D:/Data/ct_images_total"]


channels = 3

img_scale = (512, 512)

init_pipeline = [
    dict(type='LoadCategoryList', ignore_labels=['屏蔽']),
    dict(type='LoadPathList'),
    dict(type='SplitData', start=0, end=1, key='json_path_list'),
    dict(type='LoadJsonDataList'),
    dict(type='LoadLabelmeDataset'),
    dict(type='CalculateMeanAndStd', img_scale=img_scale)
]

data = dict(
    dataset_path_list=dataset_path_list,
    label_path=label_path,
)
compose = Compose(init_pipeline)
compose(data)
