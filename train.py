import numpy as np
from sklearn.model_selection import KFold
import torch
import torchvision
from torch.utils.data import DataLoader

from contrastive_loss import ContrastiveLoss
from dataset import SiameseDataset
from model import SiameseNetwork


def train(model, optimizer, criterion, dataloader):
    model.train()

    loss = []

    for img1, img2, label in dataloader:
        optimizer.zero_grad()
        img1, img2 = img1.to(device), img2.to(device)
        output1, output2 = model(img1, img2)

        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        loss.append(loss_contrastive.item())

    loss = np.array(loss)
    return loss.mean() / len(dataloader)


def save_model(model, name):
    model.eval()
    # Input to the model
    example1 = torch.randn(1, 1, 28, 28)
    example2 = torch.randn(1, 1, 28, 28)
    traced_script_module = torch.jit.trace(model.cpu(), (example1, example2))
    torch.jit.save(traced_script_module, name)


def validate(model, criterion, dataloader):
    model.eval()
    loss = []

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            output1, output2 = model(img1, img2)

            loss_contrastive = criterion(output1, output2, label)
            loss.append(loss_contrastive.item())

        loss = np.array(loss)
    return loss.mean() / len(dataloader)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

k_fold_splits = 5
batch_size = 128
lr = 0.001
epochs = 20

train_dataset = SiameseDataset(train=True, mnist=True, svhn=True, mix=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((28, 28)),
    torchvision.transforms.Grayscale()]))

k_fold = KFold(n_splits=k_fold_splits)

for fold, (train_idx, val_idx) in enumerate(k_fold.split(train_dataset)):

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=train_subsampler)
    val_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=val_subsampler)


    net = SiameseNetwork().to(device)
    contrastive_loss = ContrastiveLoss()
    adam = torch.optim.Adam(net.parameters(), lr=lr)

    rounds_without_improvement = 0
    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"--EPOCH {epoch+1}--")

        train_loss = train(model=net, optimizer=adam, criterion=contrastive_loss, dataloader=train_dataloader)
        print(f"Train loss {train_loss}")

        val_loss = validate(model=net, criterion=contrastive_loss, dataloader=val_dataloader)
        print(f"Val loss {val_loss}")

        if (val_loss < best_loss):
            best_loss = val_loss
            best_model = net
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        if (rounds_without_improvement > 3 or epoch == epochs-1):
            save_model(model=net, name=f"best_model.pt")
            break