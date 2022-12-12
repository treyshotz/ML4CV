import torch
import torchvision.transforms
from sklearn.model_selection import KFold
from torchvision.transforms import ToTensor

from non_siamese_dataset import NonSiameseDataset
from test import test_pipeline
from train import train_pipeline
from non_siamese_model import NonSiameseNetwork
from siamese_dataset import SiameseDataset, DatasetType
from siamese_model import SiameseNetwork
from transforms import AdaptiveThreshold, ToNumpy, Resize, GrayScale


class Pipelines:
    def __init__(self, k_fold_splits, batch_size, lr, epochs, transform, device, num_workers, siamese: bool):
        self.k_fold_splits = k_fold_splits
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.transform = transform
        self.device = device
        self.num_workers = num_workers

        if siamese:
            self.dataset = SiameseDataset
            self.model = SiameseNetwork
        else:
            self.dataset = NonSiameseDataset
            self.model = NonSiameseNetwork

    def mnist_svhn_mix_pipeline(self):
        print("Starting mnist-svhn-mix pipeline")
        k_fold = KFold(n_splits=self.k_fold_splits)

        train_dataset = self.dataset(train=True, dataset_type=DatasetType.MIX, transform=self.transform)
        model = train_pipeline(epochs=self.epochs, k_fold=k_fold, batch_size=self.batch_size,
                               train_dataset=train_dataset, lr=self.lr, device=self.device,
                               num_workers=self.num_workers, model=self.model)

        del train_dataset

        test_dataset = self.dataset(train=False, dataset_type=DatasetType.MIX, transform=self.transform)
        test_pipeline(test_dataset=test_dataset, batch_size=self.batch_size, device=self.device,
                      num_workers=self.num_workers,
                      model=model)

    def mnist_svhn_pipeline(self):
        print("Starting mnist-svhn pipeline")
        k_fold = KFold(n_splits=self.k_fold_splits)
        train_dataset = self.dataset(train=True, dataset_type=DatasetType.BOTH, transform=self.transform)
        model = train_pipeline(epochs=self.epochs, k_fold=k_fold, batch_size=self.batch_size,
                               train_dataset=train_dataset, lr=self.lr, device=device,
                               num_workers=self.num_workers, model=self.model)

        del train_dataset

        test_dataset = self.dataset(train=False, dataset_type=DatasetType.BOTH, transform=self.transform)
        test_pipeline(test_dataset=test_dataset, batch_size=self.batch_size, device=self.device,
                      num_workers=self.num_workers,
                      model=model)

    def mnist_pipeline(self):
        print("Starting mnist pipeline")
        k_fold = KFold(n_splits=self.k_fold_splits)
        train_dataset = self.dataset(train=True, dataset_type=DatasetType.MNIST, transform=self.transform)
        model = train_pipeline(epochs=self.epochs, k_fold=k_fold, batch_size=self.batch_size,
                               train_dataset=train_dataset, lr=self.lr, device=device,
                               num_workers=self.num_workers, model=self.model)

        del train_dataset

        test_dataset = self.dataset(train=False, dataset_type=DatasetType.MNIST, transform=self.transform)
        test_pipeline(test_dataset=test_dataset, batch_size=self.batch_size, device=self.device,
                      num_workers=self.num_workers,
                      model=model)

    def svhn_pipeline(self):
        print("Starting svhn pipeline")
        k_fold = KFold(n_splits=self.k_fold_splits)
        train_dataset = self.dataset(train=True, dataset_type=DatasetType.SVHN, transform=self.transform)
        model = train_pipeline(epochs=self.epochs, k_fold=k_fold, batch_size=self.batch_size,
                               train_dataset=train_dataset, lr=self.lr, device=device,
                               num_workers=self.num_workers, model=self.model)

        del train_dataset

        test_dataset = self.dataset(train=False, dataset_type=DatasetType.SVHN, transform=self.transform)
        test_pipeline(test_dataset=test_dataset, batch_size=self.batch_size, device=self.device,
                      num_workers=self.num_workers,
                      model=model)

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
        batch_size=1024,
        lr=0.001,
        epochs=10,
        transform=torchvision.transforms.Compose([
            ToNumpy(),
            Resize(),
            GrayScale(),
            AdaptiveThreshold(),
            ToTensor(),
        ]),
        device=device,
        num_workers=1,
        siamese=True
    )

    print("Running config 1 pipeline")
    first_config.all_pipelines()

    second_config = Pipelines(
        k_fold_splits=5,
        batch_size=1024,
        lr=0.001,
        epochs=10,
        transform=torchvision.transforms.Compose([
            ToNumpy(),
            Resize(),
            GrayScale(),
            ToTensor(),
        ]),
        device=device,
        num_workers=1,
        siamese=True
    )

    print("Running config 2 pipeline")
    second_config.all_pipelines()

    third_config = Pipelines(
        k_fold_splits=5,
        batch_size=1024,
        lr=0.001,
        epochs=10,
        transform=torchvision.transforms.Compose([
            ToNumpy(),
            Resize(),
            GrayScale(),
            AdaptiveThreshold(),
            ToTensor(),
        ]),
        device=device,
        num_workers=1,
        siamese=False
    )

    print("Running config 3 pipeline")
    third_config.mnist_svhn_mix_pipeline()
