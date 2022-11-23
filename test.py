import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataset import SiameseDataset

test_dataset = SiameseDataset(train=False, mnist=True, svhn=True, mix=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((28, 28)),
    torchvision.transforms.Grayscale()]))

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = torch.jit.load("best_model.pt").to(device)

count = 1
for img1, img2, label in test_dataloader:
    output1, output2 = model(img1, img2)

    figure = plt.figure(figsize=(8, 8))

    ax = figure.add_subplot(1, 2, 1)
    ax.set_title("Img1")
    plt.axis("off")
    plt.imshow(img1.squeeze(), cmap="gray")
    ax = figure.add_subplot(1, 2, 2)
    ax.set_title("Img2")
    plt.axis("off")
    plt.imshow(img2.squeeze(), cmap="gray")

    plt.show()

    if label == torch.FloatTensor([[0]]):
         label = "Same numbers"
    else:
        label = "Different numbers"

    print(label)
    print(F.pairwise_distance(output1, output2).item())
    print()

    count += 1
    if (count > 10):
        break