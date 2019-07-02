import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from abc import ABC, abstractmethod
from PIL import Image, ImageOps, ImageEnhance


class BaseTransform(ABC):

    def __init__(self, prob, mag):
        self.prob = prob
        self.mag = mag

    def __call__(self, img):
        return transforms.RandomApply([self.transform], self.prob)(img)

    def __repr__(self):
        return '%s(prob=%.2f, magnitude=%.2f)' % \
                (self.__class__.__name__, self.prob, self.mag)

    @abstractmethod
    def transform(self, img):
        pass


class ShearXY(BaseTransform):

    def transform(self, img):
        degrees = self.mag * 360
        t = transforms.RandomAffine(0, shear=degrees, resample=Image.BILINEAR)
        return t(img)


class TranslateXY(BaseTransform):

    def transform(self, img):
        translate = (self.mag, self.mag)
        t = transforms.RandomAffine(0, translate=translate, resample=Image.BILINEAR)
        return t(img)


class Rotate(BaseTransform):

    def transform(self, img):
        degrees = self.mag * 360
        t = transforms.RandomRotation(degrees, Image.BILINEAR)
        return t(img)


class AutoContrast(BaseTransform):

    def transform(self, img):
        cutoff = int(self.mag * 49)
        return ImageOps.autocontrast(img, cutoff=cutoff)


class Invert(BaseTransform):

    def transform(self, img):
        return ImageOps.invert(img)


class Equalize(BaseTransform):

    def transform(self, img):
        return ImageOps.equalize(img)


class Solarize(BaseTransform):

    def transform(self, img):
        threshold = (1-self.mag) * 255
        return ImageOps.solarize(img, threshold)


class Posterize(BaseTransform):

    def transform(self, img):
        bits = int((1-self.mag) * 8)
        return ImageOps.posterize(img, bits=bits)


class Contrast(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Contrast(img).enhance(factor)


class Color(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Color(img).enhance(factor)


class Brightness(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Brightness(img).enhance(factor)


class Sharpness(BaseTransform):

    def transform(self, img):
        factor = self.mag * 10
        return ImageEnhance.Sharpness(img).enhance(factor)


class Cutout(BaseTransform):

    def transform(self, img):
        n_holes = 1
        length = 24 * self.mag
        cutout_op = CutoutOp(n_holes=n_holes, length=length)
        return cutout_op(img)


class CutoutOp(object):
    """
    https://github.com/uoguelph-mlrg/Cutout

    Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        w, h = img.size

        mask = np.ones((h, w, 1), np.uint8)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h).astype(int)
            y2 = np.clip(y + self.length // 2, 0, h).astype(int)
            x1 = np.clip(x - self.length // 2, 0, w).astype(int)
            x2 = np.clip(x + self.length // 2, 0, w).astype(int)

            mask[y1: y2, x1: x2, :] = 0.

        img = mask*np.asarray(img).astype(np.uint8)
        img = Image.fromarray(mask*np.asarray(img))

        return img

