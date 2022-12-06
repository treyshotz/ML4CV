import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def test_pipeline(test_dataset, computing_device, num_workers, model):
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=num_workers)

    #Load from disk based on the model name
    if isinstance(model, str):
        model = torch.jit.load(f"{model}.pt").to(computing_device)

    model = model.to(computing_device)

    count = 1
    for img1, img2, label in test_dataloader:
        img1, img2, label = img1.to(computing_device), img2.to(computing_device), label.to(computing_device)
        output1, output2 = model(img1, img2)

        figure = plt.figure(figsize=(4, 4))
        figure.suptitle(f'Image no.{count}', fontsize=16)

        ax = figure.add_subplot(1, 1, 1)
        ax.set_title("Img1")
        plt.axis("off")
        plt.imshow(img1.cpu().squeeze(), cmap="gray")
        ax = figure.add_subplot(1, 1, 2)
        ax.set_title("Img2")
        plt.axis("off")
        plt.imshow(img2.cpu().squeeze(), cmap="gray")

        plt.show()

        print(f"Image no.{count}")
        if label.cpu() == torch.FloatTensor([[0]]):
            label = "Same numbers"
        else:
            label = "Different numbers"

        print(f"Correct label: '{label}'")
        print(F.pairwise_distance(output1, output2).item())
        print()

        count += 1
        if (count > 10):
            break
