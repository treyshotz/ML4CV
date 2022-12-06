import cv2
import cv2 as cv
import numpy as np
import torch


class ToNumpy:
    def __call__(self, sample):
        if len(sample.shape) == 2:
            sample = sample.unsqueeze(0)

        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        return sample.transpose(1, 2, 0)


class Resize:
    def __call__(self, sample):
        return cv2.resize(sample, (28, 28))


class GrayScale:
    def __call__(self, sample):
        if len(sample.shape) < 3:
            return sample
        return cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)


class EqualizeHist:
    def __call__(self, sample):
        sample = cv.normalize(sample, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        sample = cv.equalizeHist(sample)
        if len(np.where(sample.flatten() > 200)[0]) > len(np.where(sample.flatten() < 200)[0]):
            sample = 255 - sample
        return sample


class CLAHE:
    def __call__(self, sample):
        sample = cv.normalize(sample, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        clahe = cv.createCLAHE(clipLimit=3., )
        sample = clahe.apply(sample)
        if len(np.where(sample.flatten() > 200)[0]) > len(np.where(sample.flatten() < 200)[0]):
            sample = 255 - sample
        return sample


class AdaptiveThreshold:
    def __call__(self, sample):
        sample = cv.normalize(sample, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        sample = cv.adaptiveThreshold(sample, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 10)
        if len(np.where(sample.flatten() > 200)[0]) > len(np.where(sample.flatten() < 200)[0]):
            sample = 255 - sample

        return sample
