import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from contrastive_loss import ContrastiveLoss
from model import SiameseNetwork

device = ""


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


def train_pipeline(epochs, k_fold, batch_size, train_dataset, lr, computing_device, num_workers):
    global device
    device = computing_device

    global contrastive_loss
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(train_dataset)):

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=num_workers)
        val_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=num_workers)

        net = SiameseNetwork().to(computing_device)
        contrastive_loss = ContrastiveLoss()
        adam = torch.optim.Adam(net.parameters(), lr=lr)

        rounds_without_improvement = 0
        best_loss = float('inf')

        print(f"--FOLD {fold + 1}--\n")
        for epoch in range(epochs):
            print(f"--EPOCH {epoch + 1}--")

            train_loss = train(model=net, optimizer=adam, criterion=contrastive_loss, dataloader=train_dataloader)
            print(f"Train loss {train_loss}")

            val_loss = validate(model=net, criterion=contrastive_loss, dataloader=val_dataloader)
            print(f"Val loss {val_loss}")

            if (val_loss < best_loss):
                best_loss = val_loss
                save_model(model=net, name=f"fold{fold}-epoch{epoch}-transforms{train_dataset.transform}.pt")
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            if (rounds_without_improvement > 3 or epoch == epochs - 1):
                break
