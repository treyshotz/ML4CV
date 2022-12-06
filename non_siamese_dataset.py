from siamese_dataset import SiameseDataset, DatasetType
import numpy as np


class NonSiameseDataset(SiameseDataset):
    def __init__(self, train: bool, dataset_type: DatasetType, transform=None):
        super().__init__(train, dataset_type, transform)
        assert dataset_type == DatasetType.MIX

    def __getitem__(self, index):
        img1_dataset, img1_index, img2_dataset, img2_index, matching = self.pairs[index]

        if self.pre_processed:
            img1 = self.mnist_preprocessed[img1_index] if (img1_dataset == 0) else self.mnist_preprocessed[
                img2_index]
            img2 = self.svhn_preprocessed[img2_index] if (img2_dataset == 1) else self.svhn_preprocessed[
                img1_index]

        else:
            if img1_dataset == 0:
                img1 = self.mnist.data[img1_index]
                img2 = self.svhn.data[img2_index]
            else:
                img1 = self.mnist.data[img2_index]
                img2 = self.svhn.data[img1_index]
                img1_index, img2_index = img2_index, img1_index

            img1 = self.transform(img1)
            img2 = self.transform(img2)

            self.mnist_preprocessed[img1_index] = img1
            self.svhn_preprocessed[img2_index] = img2

        return img1, img2, matching

    def __len__(self):
        return super().__len__()

    def make_pairs(self):
        pairs = []
        datasets_by_label = [[np.where(self.mnist.targets == i)[0] for i in range(0, self.num_classes)],
                             [np.where(self.svhn.labels == i)[0] for i in range(0, self.num_classes)]]

        for dataset_index in range(len(datasets_by_label)):
            dataset_by_label = datasets_by_label[dataset_index]

            for label in range(len(dataset_by_label)):
                for anchor_image in dataset_by_label[label]:

                    pos_dataset_index = abs(dataset_index - 1)
                    pos_image = np.random.choice(datasets_by_label[pos_dataset_index][label])

                    pairs.append([dataset_index, anchor_image, pos_dataset_index, pos_image, 0])

                    neg_dataset_index = abs(dataset_index - 1)

                    neg_label = np.random.randint(0, self.num_classes)
                    while neg_label == label:
                        neg_label = np.random.randint(0, self.num_classes)

                    neg_image = np.random.choice(datasets_by_label[neg_dataset_index][neg_label])

                    pairs.append([dataset_index, anchor_image, neg_dataset_index, neg_image, 1])

        return pairs
