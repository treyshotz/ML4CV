from siamese_dataset import SiameseDataset, DatasetType


class NonSiameseDataset(SiameseDataset):
    def __init__(self, train: bool, dataset_type: DatasetType, transform=None):
        super().__init__(train, dataset_type, transform)
        assert dataset_type == DatasetType.BOTH

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
        return super().make_pairs()