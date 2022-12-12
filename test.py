import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from contrastive_loss import threshold_contrastive_loss


def test_pipeline(test_dataset, num_workers, model, batch_size, device):
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Load from disk based on the model name
    if isinstance(model, str):
        model = torch.jit.load(f"{model}.pt")
    model = model.to(device)
    res = [[], []]
    count = 1

    binary_threshold = 1.

    correct = 0
    total = 0.

    with torch.no_grad():
        for img1, img2, label in test_dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = model(img1, img2)

            output_label = threshold_contrastive_loss(output1, output2, binary_threshold).to(device)
            total += len(label)
            correct += (output_label == label).sum().item()

            for i in range(len(output1)):
                res[label[i]].append(F.pairwise_distance(output1[i], output2[i]).item())

            if count < 10:
                figure = plt.figure(figsize=(4, 4))
                figure.suptitle(f'Image no.{count}', fontsize=16)

                ax = figure.add_subplot(1, 2, 1)
                ax.set_title("Img1")
                plt.axis("off")
                plt.imshow(img1[0].cpu().squeeze(), cmap="gray")
                ax = figure.add_subplot(1, 2, 2)
                ax.set_title("Img2")
                plt.axis("off")
                plt.imshow(img2[0].cpu().squeeze(), cmap="gray")

                plt.show()

                print(f"Image no.{count}")
                if label[0].cpu() == torch.FloatTensor([[0]]):
                    caption = "Same numbers"
                else:
                    caption = "Different numbers"

                print(f"Correct label: '{caption}'")
                print(F.pairwise_distance(output1[0], output2[0]).item())
                print()

            count += 1

        print("\nImages with same number")
        print(f"Mean: {torch.mean(torch.tensor(res[0]))}")
        print(f"Std: {torch.std(torch.tensor(res[0]))}\n")

        print("Images with different number")
        print(f"Mean: {torch.mean(torch.tensor(res[1]))}")
        print(f"Std: {torch.std(torch.tensor(res[1]))}\n")

        print(f"Accuracy {correct / total}")
        fig = plt.figure(1, figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.boxplot(res)
        plt.show()
