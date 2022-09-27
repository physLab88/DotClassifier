import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# ======================== TRAINING SETTINGS ========================
BATCH_SIZE = 16
LEARNING_RATE = 1E-2
EPOCHS = 10

# ======================= SETTING UP DATASET ========================
data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),transforms.RandomRotation(30),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])]),
                   'valid': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),
                   'test': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
}

img_datasets = {
    'train': datasets.Flowers102(root="data", split="train", download=True, transform=data_transforms['train']),
    'test': datasets.Flowers102(root="data", split="test", download=True, transform=data_transforms['test']),
    'valid': datasets.Flowers102(root="data", split="val", download=True, transform=data_transforms['valid'])}

img_dataloaders = {key: DataLoader(img_datasets[key], batch_size=BATCH_SIZE, shuffle=True) for key in img_datasets}


def lookAtData(n):
    for i in range(n):
        images, labels = next(iter(img_dataloaders["train"]))
        print(images.shape)
        plt.imshow(images[np.random.randint(0, BATCH_SIZE),np.random.randint(0, 3)])
        plt.title(labels[1])
        plt.show()


# ====================== BUILDING THE NET ======================
resnet50 = models.resnet50(weights=True)
# print(resnet50._modules.keys())
# print(resnet50._modules['fc'])
last_layer = 'fc'
temp_in = resnet50._modules[last_layer].in_features
temp_out = 102
resnet50._modules[last_layer] = nn.Linear(temp_in, temp_out)
resnet50 = resnet50.to(device)

# ==================== TRAINING AND TESTING ====================
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        i = 0
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            i += 1
            #print('v %s' % i)

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return(100*correct)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    i=0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
        #print('t %s' % i)

        if batch % 12 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


loss_fn = nn.CrossEntropyLoss()  # KLDivLoss()
optimizer = torch.optim.SGD(resnet50.parameters(), lr=LEARNING_RATE)


# ============================ MAIN ============================
def main():
    success = []
    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(img_dataloaders['test'], resnet50, loss_fn, optimizer)
        success.append(test_loop(img_dataloaders['train'], resnet50, loss_fn))
    print("Done!")
    plt.plot(np.arange(len(success)) + 1, np.array(success), 'ko')
    plt.title('Flower ResNet50 KLDivLoss')
    plt.xlabel('epoche')
    plt.ylabel('success')
    plt.show()

    torch.save(resnet50, 'resnet_flower.pt')


if __name__ == '__main__':
    main()


