import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from contrastive_loss import ContrastiveLoss

device = ""


def train(model, optimizer, criterion, dataloader):
    model.train()

    loss = []

    for img1, img2, label in dataloader:
        optimizer.zero_grad()

        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
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
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = model(img1, img2)

            loss_contrastive = criterion(output1, output2, label)
            loss.append(loss_contrastive.item())

        loss = np.array(loss)
    return loss.mean() / len(dataloader)


def train_pipeline(epochs, k_fold, batch_size, train_dataset, lr, computing_device, num_workers, model):
    global device
    best_model = ""
    device = computing_device

    global contrastive_loss
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(train_dataset)):

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler,
                                      num_workers=num_workers)
        val_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler,
                                    num_workers=num_workers)

        net = model.to(computing_device)
        contrastive_loss = ContrastiveLoss()
        adam = torch.optim.Adam(net.parameters(), lr=lr)

        rounds_without_improvement = 0
        best_loss = float('inf')
        best_epoch = 0

        print(f"--FOLD {fold + 1}--\n")
        for epoch in range(epochs):
            print(f"--EPOCH {epoch + 1}--")

            train_loss = train(model=net, optimizer=adam, criterion=contrastive_loss, dataloader=train_dataloader)
            train_dataset.pre_processed = True
            print(f"Train loss {train_loss}")

            val_loss = validate(model=net, criterion=contrastive_loss, dataloader=val_dataloader)
            print(f"Val loss {val_loss}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_model = copy.deepcopy(net)
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            if rounds_without_improvement > 5 or epoch == epochs - 1:
                save_model(model=best_model,
                           name=f"fold{fold}-epoch{best_epoch + 1}-transforms{random.randint(0, 10000)}.pt")
                break

    return best_model
