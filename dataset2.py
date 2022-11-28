import time
from enum import Enum

import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from transforms import Resize, GrayScale, ToNumpy, EqualizeHist, AdaptiveThreshold


class DatasetType(Enum):
    MNIST = 1,
    SVHN = 2,
    BOTH = 3,
    MIX = 4


class SiameseDataset(Dataset):
    def __init__(self, train: bool, dataset_type: DatasetType, transform=None):
        self.mnist = None
        self.svhn = None
        self.dataset_type = dataset_type
        self.transform = transform

        self.num_classes = 10

        if self.dataset_type != DatasetType.SVHN:
            self.mnist = torchvision.datasets.MNIST("files", train=train, download=True)

        if self.dataset_type != DatasetType.MNIST:
            self.svhn = torchvision.datasets.SVHN(root="data", split="train" if train else "test", download=True)

        self.pairs = self.make_pairs()

    def __getitem__(self, index):
        img1_dataset, img1_index, img2_dataset, img2_index, matching = self.pairs[index]

        img1 = self.mnist.data[img1_index] if (img1_dataset == 0) else self.svhn.data[img1_index]
        img2 = self.mnist.data[img2_index] if (img2_dataset == 0) else self.svhn.data[img2_index]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

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


t = torchvision.transforms.Compose([
    ToNumpy(),
    Resize(),
    GrayScale(),
    EqualizeHist(),
    AdaptiveThreshold(),
    ToTensor(),
])


d = SiameseDataset(train=True, dataset_type=DatasetType.MIX, transform=t)

dl = DataLoader(d, shuffle=True, batch_size=2)
start = time.time()
batch = dl.__iter__().next()
end = time.time()

figure = plt.figure(figsize=(12, 8))
for i in range(len(batch[0])):
    ax = figure.add_subplot(2, 2, i*2+1)
    ax.set_title("Img")
    plt.axis("off")
    plt.imshow(batch[0][i].squeeze(), cmap="gray")

    ax = figure.add_subplot(2, 2, i*2+2)
    ax.set_title("Img2")
    plt.axis("off")
    plt.imshow(batch[1][i].squeeze(), cmap="gray")

    print(batch[2][i])


plt.show()
print(end - start)
print()
