from numpy import random
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class RandomAdd:

    def __init__(
            self, channel=1, random_range=2000, ratio=0.5, *args, **kwargs):
        self.channel = channel
        self.random_range = random_range
        self.ratio = ratio

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if random.random() <= self.ratio:
                img_channel = img[:, :, self.channel]
                np.add(
                    img_channel[img_channel > 0],
                    random.randint(-self.random_range, self.random_range),
                    out=img_channel[img_channel > 0],
                    casting="unsafe")
                # img_channel[img_channel > 0] += random.randint(
                #     -self.random_range, self.random_range)
            results[key] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class SubMean:

    def __init__(self, channel=1, *args, **kwargs):
        self.channel = channel

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img = img.astype(np.float32)
            img_channel = img[:, :, self.channel]
            img_channel[img_channel > 0] -= img_channel[img_channel > 0].mean()
            results[key] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
