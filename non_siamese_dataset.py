import numpy as np
import torchvision
from torch.utils.data import Dataset

from siamese_dataset import DatasetType


class NonSiameseDataset(Dataset):
    def __init__(self, train: bool, dataset_type: DatasetType, transform=None):
        self.transform = transform

        self.num_classes = 10

        self.mnist = torchvision.datasets.MNIST("files", train=train, download=True)
        print("Preprocessing MNIST")
        self.mnist_preprocessed = list(map(self.transform, self.mnist.data))
        print("MNIST preprocessed")

        self.svhn = torchvision.datasets.SVHN(root="data", split="extra" if train else "test", download=True)
        print("Preprocessing SVHN")
        self.svhn_preprocessed = list(map(self.transform, self.svhn.data))
        print("SVHN preprocessed")

        self.pairs = self.make_pairs()

    def __getitem__(self, index):
        img1_index, img2_index, matching = self.pairs[index]

        img1 = self.mnist_preprocessed[img1_index]
        img2 = self.svhn_preprocessed[img2_index]

        return img1, img2, matching

    def __len__(self):
        return len(self.pairs)

    def make_pairs(self):
        pairs = []
        datasets_by_label = [[np.where(self.mnist.targets == i)[0] for i in range(0, self.num_classes)],
                             [np.where(self.svhn.labels == i)[0] for i in range(0, self.num_classes)]]

        for i in range(0, len(datasets_by_label[1])):
            datasets_by_label[1][i] = datasets_by_label[1][i][0:6000]

        for dataset_index in range(len(datasets_by_label)):
            dataset_by_label = datasets_by_label[dataset_index]

            for label in range(len(dataset_by_label)):
                for anchor_image in dataset_by_label[label]:

                    other_dataset = abs(dataset_index - 1)
                    pos_image = np.random.choice(datasets_by_label[other_dataset][label])

                    neg_label = np.random.randint(0, self.num_classes)
                    while neg_label == label:
                        neg_label = np.random.randint(0, self.num_classes)

                    neg_image = np.random.choice(datasets_by_label[other_dataset][neg_label])

                    if dataset_index == 0:
                        pairs.append([anchor_image, pos_image, 0])
                        pairs.append([anchor_image, neg_image, 1])
                    else:
                        pairs.append([pos_image, anchor_image, 0])
                        pairs.append([neg_label, anchor_image, 1])

        return pairs
