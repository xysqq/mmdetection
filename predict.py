import cv2
import matplotlib.pyplot as plt

from mmdet.apis import init_detector, inference_detector,show_result_pyplot

config_file = 'configs2/uw_madison_gi_tract/胃肠道图像分割.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
checkpoint_file = 'work_dirs/胃肠道图像分割/latest.pth'
device = 'cuda:0'
img = r'./train\case101\case101_day20\scans\slice_0066_266_266_1.50_1.50.png'

# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
result = inference_detector(model, img)
#
show_result_pyplot(model, img, result)
