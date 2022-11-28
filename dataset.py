import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from transforms import ResizeGrayscale


class SiameseDataset(Dataset):
    def __init__(self, train: bool, transforms, mnist=False, svhn=False, mix=False, ):
        self.mnist_dataset = None
        self.svhn_dataset = None
        self.transforms = transforms
        self.pre_processed = False

        if mnist:
            self.mnist_dataset = torchvision.datasets.MNIST("files", train=train, download=True,
                                                            transform=torchvision.transforms.Compose([
                                                                                                         torchvision.transforms.ToTensor(),
                                                                                                     ] + self.transforms))
            self.mnist_preprocessed = torch.zeros((len(self.mnist_dataset.data), 1, 28, 28))

        if svhn:
            if train:
                split = "train"
            else:
                split = "test"

            self.svhn_dataset = torchvision.datasets.SVHN(root="data", split=split, download=True,
                                                          transform=torchvision.transforms.Compose([
                                                                                                       torchvision.transforms.ToTensor(),
                                                                                                       ResizeGrayscale(),
                                                                                                   ] + self.transforms))
            self.svhn_preprocessed = torch.zeros((len(self.svhn_dataset.data), 1, 28, 28))
        # used to prepare the labels and images path
        self.pairs = make_pairs(mix, self.mnist_dataset, self.svhn_dataset)

        self.dataset = [self.mnist_dataset] + [self.svhn_dataset]
        self.pre_processed_dataset = [self.mnist_preprocessed] + [self.svhn_preprocessed]

    def __getitem__(self, index):

        img1_dataset, img1_index = self.pairs[index][0]
        img2_dataset, img2_index = self.pairs[index][1]

        matching = self.pairs[index][2]

        if self.pre_processed:
            return self.pre_processed_dataset[img1_dataset][img1_index], \
                   self.pre_processed_dataset[img2_dataset][img2_index], matching
        else:
            pre_img1 = self.dataset[img1_dataset].__getitem__(img1_index)[0]
            pre_img2 = self.dataset[img2_dataset].__getitem__(img2_index)[0]

            self.pre_processed_dataset[img1_dataset][img1_index] = pre_img1
            self.pre_processed_dataset[img2_dataset][img2_index] = pre_img2

            return pre_img1, pre_img2, matching

    def __len__(self):
        return len(self.pairs)


def make_pairs(mix, mnist=None, svhn=None):
    pairs = []

    num_classes = 10

    if mix and mnist and svhn:
        return mix_pairs(mnist, num_classes, svhn)
    if svhn:
        svhn_labels = svhn.labels
        svhn_idx = [np.where(svhn_labels == i)[0] for i in range(0, num_classes)]
        dataset_pos = 1

        for anchor_idx in range(len(svhn_labels)):
            label = svhn_labels[anchor_idx]

            pos_idx = np.random.choice(svhn_idx[label])

            pairs.append([(dataset_pos, anchor_idx), (dataset_pos, pos_idx), 0])

            negative_label = np.random.randint(0, num_classes)
            while negative_label == label:
                negative_label = np.random.randint(0, num_classes)

            neg_idx = np.random.choice(svhn_idx[negative_label])

            pairs.append([(dataset_pos, anchor_idx), (dataset_pos, neg_idx), 1])
    if mnist:
        mnist_labels = mnist.targets
        mnist_idx = [np.where(mnist_labels == i)[0] for i in range(0, num_classes)]
        dataset_pos = 0

        for anchor_idx in range(len(mnist_labels)):
            label = mnist_labels[anchor_idx]

            pos_idx = np.random.choice(mnist_idx[label])

            pairs.append([(dataset_pos, anchor_idx), (dataset_pos, pos_idx), 0])

            negative_label = np.random.randint(0, num_classes)
            while negative_label == label:
                negative_label = np.random.randint(0, num_classes)

            neg_idx = np.random.choice(mnist_idx[negative_label])

            pairs.append([(dataset_pos, anchor_idx), (dataset_pos, neg_idx), 1])

    return pairs


def add_pairs_mix(dataset_labels, dataset_pos, mnist_idx, svhn_idx, num_classes):
    pairs = []
    for anchor_idx in range(len(dataset_labels)):
        mnist_label = dataset_labels[anchor_idx]

        dataset_choice = np.random.randint(0, 2)
        # 0 = MNIST, 1 = SVHN

        if dataset_choice == 0:
            pos_idx = np.random.choice(mnist_idx[mnist_label])
        else:
            pos_idx = np.random.choice(svhn_idx[mnist_label])

        pairs.append([(dataset_pos, anchor_idx), (dataset_choice, pos_idx), 0])

        negative_label = np.random.randint(0, num_classes)
        while negative_label == mnist_label:
            negative_label = np.random.randint(0, num_classes)

        dataset_choice = np.random.randint(0, 2)

        if dataset_choice == 0:
            neg_idx = np.random.choice(mnist_idx[negative_label])
        else:
            neg_idx = np.random.choice(svhn_idx[negative_label])

        pairs.append([(dataset_pos, anchor_idx), (dataset_choice, neg_idx), 1])
    return pairs


def mix_pairs(mnist, num_classes, svhn):
    pairs = []
    ### Add mixing of datasets
    mnist_labels = mnist.targets
    svhn_labels = svhn.labels
    mnist_idx = [np.where(mnist_labels == i)[0] for i in range(0, num_classes)]
    svhn_idx = [np.where(svhn_labels == i)[0] for i in range(0, num_classes)]
    mnist_dataset_pos = 0
    svhn_dataset_pos = 1

    pairs = pairs + add_pairs_mix(mnist_labels, mnist_dataset_pos, mnist_idx, svhn_idx, num_classes)

    pairs = pairs + add_pairs_mix(svhn_labels, svhn_dataset_pos, mnist_idx, svhn_idx, num_classes)

    return pairs
