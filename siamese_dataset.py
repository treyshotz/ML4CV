from enum import Enum

import numpy as np
import torchvision
from torch.utils.data import Dataset


class DatasetType(Enum):
    MNIST = 1,
    SVHN = 2,
    BOTH = 3,
    MIX = 4


class SiameseDataset(Dataset):
    def __init__(self, train: bool, dataset_type: DatasetType, transform=None):
        self.dataset_type = dataset_type
        self.transform = transform
        self.train = train

        self.num_classes = 10

        if self.dataset_type != DatasetType.SVHN:
            self.mnist = torchvision.datasets.MNIST("files", train=train, download=True)
            print("Preprocessing MNIST")
            self.mnist_preprocessed = list(map(self.transform, self.mnist.data))
            print("MNIST preprocessed")

        if self.dataset_type != DatasetType.MNIST:
            self.svhn = torchvision.datasets.SVHN(root="data", split="extra" if train else "test", download=True)
            print("Preprocessing SVHN")
            self.svhn_preprocessed = list(map(self.transform, self.svhn.data))
            print("SVHN preprocessed")

        self.pairs = self.make_pairs()

    def __getitem__(self, index):
        img1_dataset, img1_index, img2_dataset, img2_index, matching = self.pairs[index]

        img1 = self.mnist_preprocessed[img1_index] if (img1_dataset == 0) else self.svhn_preprocessed[img1_index]
        img2 = self.mnist_preprocessed[img2_index] if (img2_dataset == 0) else self.svhn_preprocessed[img2_index]

        return img1, img2, matching

    def __len__(self):
        return len(self.pairs)

    def make_pairs(self):
        pairs = []

        if self.dataset_type == DatasetType.MNIST or self.dataset_type == DatasetType.BOTH:
            mnist_by_label = [np.where(self.mnist.targets == i)[0] for i in range(0, self.num_classes)]

            for label in range(len(mnist_by_label)):
                for anchor_image in mnist_by_label[label]:

                    pos_image = np.random.choice(mnist_by_label[label])
                    pairs.append([0, anchor_image, 0, pos_image, 0])

                    neg_label = np.random.randint(0, self.num_classes)
                    while neg_label == label:
                        neg_label = np.random.randint(0, self.num_classes)

                    neg_image = np.random.choice(mnist_by_label[neg_label])
                    pairs.append([0, anchor_image, 0, neg_image, 1])

        if self.dataset_type == DatasetType.SVHN or self.dataset_type == DatasetType.BOTH:
            svhn_by_label = [np.where(self.svhn.labels == i)[0] for i in range(0, self.num_classes)]
            for i in range(0, self.num_classes):
                limit = 6000 if self.train is True else 1000
                svhn_by_label[i] = svhn_by_label[i][0:limit]

            for label in range(len(svhn_by_label)):
                for anchor_image in svhn_by_label[label]:

                    pos_image = np.random.choice(svhn_by_label[label])

                    pairs.append([1, anchor_image, 1, pos_image, 0])

                    neg_label = np.random.randint(0, self.num_classes)
                    while neg_label == label:
                        neg_label = np.random.randint(0, self.num_classes)

                    neg_image = np.random.choice(svhn_by_label[neg_label])

                    pairs.append([1, anchor_image, 1, neg_image, 1])

        if self.dataset_type == DatasetType.MIX:
            datasets_by_label = [[np.where(self.mnist.targets == i)[0] for i in range(0, self.num_classes)],
                                 [np.where(self.svhn.labels == i)[0] for i in range(0, self.num_classes)]]
            for i in range(0, len(datasets_by_label[1])):
                limit = 6000 if self.train is True else 1000
                datasets_by_label[1][i] = datasets_by_label[1][i][0:limit]

            for dataset_index in range(len(datasets_by_label)):
                dataset_by_label = datasets_by_label[dataset_index]

                for label in range(len(dataset_by_label)):
                    for anchor_image in dataset_by_label[label]:

                        pos_dataset_index = np.random.randint(0, 2)
                        pos_image = np.random.choice(datasets_by_label[pos_dataset_index][label])

                        pairs.append([dataset_index, anchor_image, pos_dataset_index, pos_image, 0])

                        neg_dataset_index = np.random.randint(0, 2)
                        neg_label = np.random.randint(0, self.num_classes)
                        while neg_label == label:
                            neg_label = np.random.randint(0, self.num_classes)

                        neg_image = np.random.choice(datasets_by_label[neg_dataset_index][neg_label])

                        pairs.append([dataset_index, anchor_image, neg_dataset_index, neg_image, 1])

        return pairs
