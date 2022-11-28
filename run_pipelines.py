from multiprocessing import freeze_support

import torch
import torchvision.transforms
from sklearn.model_selection import KFold
from torchvision.transforms import ToTensor

import test
import train
from dataset2 import SiameseDataset, DatasetType
from transforms import AdaptiveThreshold, EqualizeHist, ToNumpy, Resize, GrayScale


class Pipelines:
    def __init__(self, k_fold_splits, batch_size, lr, epochs, transform, device, num_workers):
        self.k_fold_splits = k_fold_splits
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.transform = transform
        self.device = device
        self.num_workers = num_workers

    def mnist_svhn_mix_pipeline(self):
        k_fold = KFold(n_splits=self.k_fold_splits)
        train_dataset = SiameseDataset(train=True, dataset_type=DatasetType.MIX, transform=self.transform)
        train.train_pipeline(epochs=self.epochs, k_fold=k_fold, batch_size=self.batch_size, train_dataset=train_dataset,
                             lr=self.lr, computing_device=self.device, num_workers=self.num_workers)

        del train_dataset

        test_dataset = SiameseDataset(train=False, dataset_type=DatasetType.MIX, transform=self.transform)
        test.test_pipeline(test_dataset=test_dataset, computing_device=self.device, num_workers=self.num_workers)

    def mnist_svhn_pipeline(self):
        k_fold = KFold(n_splits=self.k_fold_splits)
        train_dataset = SiameseDataset(train=True, dataset_type=DatasetType.BOTH, transform=self.transform)
        train.train_pipeline(epochs=self.epochs, k_fold=k_fold, batch_size=self.batch_size, train_dataset=train_dataset,
                             lr=self.lr, computing_device=device, num_workers=self.num_workers)

        del train_dataset

        test_dataset = SiameseDataset(train=False, dataset_type=DatasetType.BOTH, transform=self.transform)
        test.test_pipeline(test_dataset=test_dataset, computing_device=device, num_workers=self.num_workers)

    def mnist_pipeline(self):
        k_fold = KFold(n_splits=self.k_fold_splits)
        train_dataset = SiameseDataset(train=True, dataset_type=DatasetType.MNIST, transform=self.transform)
        train.train_pipeline(epochs=self.epochs, k_fold=k_fold, batch_size=self.batch_size, train_dataset=train_dataset,
                             lr=self.lr, computing_device=device, num_workers=self.num_workers)

        del train_dataset

        test_dataset = SiameseDataset(train=False, dataset_type=DatasetType.MNIST, transform=self.transform)
        test.test_pipeline(test_dataset=test_dataset, computing_device=device, num_workers=self.num_workers)

    def svhn_pipeline(self):
        k_fold = KFold(n_splits=self.k_fold_splits)
        train_dataset = SiameseDataset(train=True, dataset_type=DatasetType.SVHN, transform=self.transform)
        train.train_pipeline(epochs=self.epochs, k_fold=k_fold, batch_size=self.batch_size, train_dataset=train_dataset,
                             lr=self.lr, computing_device=device, num_workers=self.num_workers)

        del train_dataset

        test_dataset = SiameseDataset(train=False, dataset_type=DatasetType.SVHN, transform=self.transform)
        test.test_pipeline(test_dataset=test_dataset, computing_device=device, num_workers=self.num_workers)

    def all_pipelines(self):
        self.mnist_svhn_mix_pipeline()
        self.mnist_svhn_pipeline()
        self.mnist_pipeline()
        self.svhn_pipeline()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    first_config = Pipelines(
        k_fold_splits=5,
        batch_size=1028,
        lr=0.001,
        epochs=20,
        transform=torchvision.transforms.Compose([
            ToNumpy(),
            Resize(),
            GrayScale(),
            AdaptiveThreshold(),
            ToTensor(),
        ]),
        device=device,
        num_workers=1,
    )

    first_config.all_pipelines()

    second_config = Pipelines(
        k_fold_splits=5,
        batch_size=1028,
        lr=0.001,
        epochs=20,
        transform=torchvision.transforms.Compose([
            ToNumpy(),
            Resize(),
            GrayScale(),
            ToTensor(),
            EqualizeHist()
        ]),
        device=device,
        num_workers=1,
    )
