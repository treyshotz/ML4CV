import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as T



class ResizeGrayscale:
    def __call__(self, sample):
        reshaped = T.Resize((28, 28))(sample)
        gray_reshaped = T.Grayscale()(reshaped)
        return 1 - gray_reshaped


class EqualizeHist:
    def __call__(self, sample):
        sample = sample.numpy()
        sample = cv.normalize(sample, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        sample = sample.astype(np.uint8)
        sample = cv.equalizeHist(sample[0])
        if len(np.where(sample.flatten() > 200)[0]) > len(np.where(sample.flatten() < 200)[0]):
            sample = 255 - sample
        sample = sample / 255
        return torch.from_numpy(sample).unsqueeze(0)


class CLAHE:
    def __call__(self, sample):
        sample = sample.numpy()
        sample = cv.normalize(sample, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        sample = sample.astype(np.uint8)
        clahe = cv.createCLAHE(clipLimit=3., )
        sample = clahe.apply(sample)
        if len(np.where(sample.flatten() > 200)[0]) > len(np.where(sample.flatten() < 200)[0]):
            sample = 255 - sample
        sample = sample / 255
        return torch.from_numpy(sample).unsqueeze(0)


class AdaptiveThreshold:
    def __call__(self, sample):
        sample = sample.numpy()
        sample = cv.normalize(sample, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        sample = sample.astype(np.uint8)
        sample = cv.adaptiveThreshold(sample[0], 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 0)
        if len(np.where(sample.flatten() > 200)[0]) > len(np.where(sample.flatten() < 200)[0]):
            sample = 255 - sample
        sample = torch.from_numpy(sample).unsqueeze(0)
        sample = sample / 255
        return sample
