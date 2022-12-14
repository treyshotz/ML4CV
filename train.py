import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from contrastive_loss import ContrastiveLoss, threshold_contrastive_loss


def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage. Experienced this when using k-fold
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def train(model, optimizer, criterion, dataloader, device):
    model.train()

    # If image pair has distance above this value they are considered as dissimilar
    binary_threshold = 1.

    loss = []
    correct = 0
    total = 0.
    for img1, img2, label in dataloader:
        optimizer.zero_grad()

        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        output1, output2 = model(img1, img2)

        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        loss.append(loss_contrastive.item())

        output_label = threshold_contrastive_loss(output1, output2, binary_threshold).to(device)
        total += len(label)
        correct += (output_label == label).sum().item()

    loss = np.array(loss)
    return loss.mean() / len(dataloader), correct / total


def save_model(model, name):
    model.eval()
    # Input to the model
    example1 = torch.randn(1, 1, 28, 28)
    example2 = torch.randn(1, 1, 28, 28)
    traced_script_module = torch.jit.trace(model.cpu(), (example1, example2))
    torch.jit.save(traced_script_module, name)
    print(f"Saved model with name: {name}")


def validate(model, criterion, dataloader, device):
    model.eval()

    # If image pair has distance above this value they are considered as dissimilar
    binary_threshold = 1.

    loss = []
    correct = 0
    total = 0.
    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = model(img1, img2)

            loss_contrastive = criterion(output1, output2, label)
            loss.append(loss_contrastive.item())

            output_label = threshold_contrastive_loss(output1, output2, binary_threshold).to(device)
            total += len(label)
            correct += (output_label == label).sum().item()

        loss = np.array(loss)
    return loss.mean() / len(dataloader), correct / total


def train_pipeline(epochs, k_fold, batch_size, train_dataset, lr, device, num_workers, model):
    contrastive_loss = ContrastiveLoss()
    best_model = ""

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(train_dataset)):

        # Creating val loader and train loader based on current fold split
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler,
                                      num_workers=num_workers)
        val_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler,
                                    num_workers=num_workers)

        net = model()
        net.apply(reset_weights)
        net = net.to(device)

        adam = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)

        rounds_without_improvement = 0
        best_loss = float('inf')
        best_epoch = 0

        print(f"--FOLD {fold + 1}--\n")
        for epoch in range(epochs):
            print(f"--EPOCH {epoch + 1}--")

            train_loss, train_acc = train(model=net, optimizer=adam, criterion=contrastive_loss, device=device,
                                          dataloader=train_dataloader)
            print(f"Train loss {train_loss:.5f}, Train acc {train_acc:.5f}")

            val_loss, val_acc = validate(model=net, criterion=contrastive_loss, device=device,
                                         dataloader=val_dataloader)
            print(f"Val loss {val_loss:.5f}, Val acc {val_acc:.5f}")

            # Saving best model so far
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_model = copy.deepcopy(net)
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            # Early stopping if model cease to improve
            if rounds_without_improvement > 3:
                break

        save_model(model=best_model,
                   name=f"fold{fold + 1}-epoch{best_epoch + 1}-transforms{random.randint(0, 10000)}.pt")

    return best_model
