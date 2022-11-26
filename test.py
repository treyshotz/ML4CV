import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def test_pipeline(test_dataset, computing_device):
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    model = torch.jit.load("fold0epoch29.pt").to(computing_device)
    count = 1
    for img1, img2, label in test_dataloader:
        output1, output2 = model(img1, img2)

        figure = plt.figure(figsize=(8, 8))
        figure.suptitle(f'Image no.{count}', fontsize=16)

        ax = figure.add_subplot(1, 2, 1)
        ax.set_title("Img1")
        plt.axis("off")
        plt.imshow(img1.squeeze(), cmap="gray")
        ax = figure.add_subplot(1, 2, 2)
        ax.set_title("Img2")
        plt.axis("off")
        plt.imshow(img2.squeeze(), cmap="gray")

        plt.show()

        print(f"Image no.{count}")
        if label == torch.FloatTensor([[0]]):
            label = "Same numbers"
        else:
            label = "Different numbers"

        print(f"Correct label: '{label}'")
        print(F.pairwise_distance(output1, output2).item())
        print()

        count += 1
        if (count > 10):
            break
