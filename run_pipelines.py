import torch
from sklearn.model_selection import KFold

from dataset import SiameseDataset
import train
import test
from transforms import AdaptiveThreshold, CLAHE, EqualizeHist


class Pipelines:
    def __init__(self, k_fold_splits, batch_size, lr, epochs, transforms, device):
        self.k_fold_splits = k_fold_splits
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.transforms = transforms
        self.device = device

    def mnist_svhn_mix_pipeline(self):
        k_fold = KFold(n_splits=self.k_fold_splits)
        train_dataset = SiameseDataset(train=True, mnist=True, svhn=True, mix=True, transforms=self.transforms)
        train.train_pipeline(epochs=self.epochs, k_fold=k_fold, batch_size=self.batch_size, train_dataset=train_dataset,
                             lr=self.lr,
                             computing_device=self.device)

        del train_dataset

        test_dataset = SiameseDataset(train=False, mnist=True, svhn=True, mix=True, transforms=self.transforms)
        test.test_pipeline(test_dataset=test_dataset, computing_device=self.device)

    def mnist_svhn_pipeline(self):
        k_fold = KFold(n_splits=self.k_fold_splits)
        train_dataset = SiameseDataset(train=True, mnist=True, svhn=True, mix=False, transforms=self.transforms)
        train.train_pipeline(epochs=self.epochs, k_fold=k_fold, batch_size=self.batch_size, train_dataset=train_dataset,
                             lr=self.lr,
                             computing_device=device)

        del train_dataset

        test_dataset = SiameseDataset(train=False, mnist=True, svhn=True, mix=False, transforms=self.transforms)
        test.test_pipeline(test_dataset=test_dataset, computing_device=device)

    def mnist_pipeline(self):
        k_fold = KFold(n_splits=self.k_fold_splits)
        train_dataset = SiameseDataset(train=True, mnist=True, svhn=False, mix=False, transforms=self.transforms)
        train.train_pipeline(epochs=self.epochs, k_fold=k_fold, batch_size=self.batch_size, train_dataset=train_dataset,
                             lr=self.lr,
                             computing_device=device)

        del train_dataset

        test_dataset = SiameseDataset(train=False, mnist=True, svhn=False, mix=False, transforms=self.transforms)
        test.test_pipeline(test_dataset=test_dataset, computing_device=device)

    def svhn_pipeline(self):
        k_fold = KFold(n_splits=self.k_fold_splits)
        train_dataset = SiameseDataset(train=True, mnist=False, svhn=True, mix=False, transforms=self.transforms)
        train.train_pipeline(epochs=self.epochs, k_fold=k_fold, batch_size=self.batch_size, train_dataset=train_dataset,
                             lr=self.lr,
                             computing_device=device)

        del train_dataset

        test_dataset = SiameseDataset(train=False, mnist=False, svhn=True, mix=False, transforms=self.transforms)
        test.test_pipeline(test_dataset=test_dataset, computing_device=device)

    def all_pipelines(self):
        self.mnist_svhn_mix_pipeline()
        self.mnist_svhn_pipeline()
        self.mnist_pipeline()
        self.svhn_pipeline()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

first_config = Pipelines(
    k_fold_splits=5,
    batch_size=128,
    lr=0.001,
    epochs=20,
    transforms=[
        AdaptiveThreshold()
    ],
    device=device
)

first_config.all_pipelines()

second_config = Pipelines(
    k_fold_splits=5,
    batch_size=128,
    lr=0.001,
    epochs=20,
    transforms=[
        EqualizeHist()
    ],
    device=device)
