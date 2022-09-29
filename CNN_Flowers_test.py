import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from torchvision import models
from datetime import datetime
import cronos as cron
import wandb

# ===================== GLOBAL VARIABLES =====================
RUN_NAME = ''
ID = ''
NETWORK_DIRECTORY = "Networks/Flower102/"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

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


def lookAtData(n):
    # broken function
    for i in range(n):
        images, labels = next(iter(img_dataloaders["train"]))
        print(images.shape)
        plt.imshow(images[np.random.randint(0, BATCH_SIZE),np.random.randint(0, 3)])
        plt.title(labels[1])
        plt.show()

# ==================== TRAINING AND TESTING ====================
def test_loop(dataloader, model, loss_fn):
    print("=========== Evaluating model ===========")
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
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    i=0
    cron.start('train_print')
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

        if cron.t('train_print') >= 5.0:
            cron.start('train_print')
            loss, current = loss.item(), batch * len(X)
            temp = current/size
            bar = '[' + int(temp*30)*'#' + int((1-temp)*30)*'-' + ']'
            print("loss: {loss:.5f}  {bar:>} {perc:.2%}".format(loss=loss, bar=bar, perc=temp))
    cron.stop('train_print')


def basicTrainingSequence(model, loss_fn, optimizer, train_dataloader, test_dataloader, numEpoch, filename=None):
    # TODO initial assestement
    # TODO continue run ???
    # ------------->>> training loop
    for E in range(numEpoch):
        print("Starting Epoch %s \n ========================================" % (E+1))
        train_loop(train_dataloader, model, loss_fn, optimizer)
        accu, loss = test_loop(test_dataloader, model, loss_fn)

        wandb.log({"loss": loss, "accuracy": accu, "epoch": E + 1})
        wandb.config.update({"epochs": E + 1}, allow_val_change=True)
        # ------------>>> make a checkpoint
        # ---> save report
        # ---> save optim
        # ---> save model
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(NETWORK_DIRECTORY + RUN_NAME + '_' + ID + '.pt')  # Save
    # ------------>>> final save
    # ---> save report
    # ---> save model
    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save(NETWORK_DIRECTORY + RUN_NAME + '_' + ID + '.pt')  # Save


# ====================== BUILDING THE NET ======================
model = models.resnet50(weights=False)
# print(model._modules.keys())
# print(model._modules['fc'])
last_layer = 'fc'
temp_in = model._modules[last_layer].in_features
temp_out = 102
model._modules[last_layer] = nn.Linear(temp_in, temp_out)
model = model.to(device)


# ============================ MAIN ============================
def main():
    # TODO load models
    # TODO randomise weights
    global ID, RUN_NAME
    ID = wandb.util.generate_id()
    # ID =
    configs = {
        "learning_rate": 7E-2,
        "epochs": 10,
        "batch_size": 16,
        "architecture": "ResNet50",
        "pretrained": False,
        "loss_fn": "CrossEntropyLoss",
        "optimiser": "SGD",
    }
    img_dataloaders = {key: DataLoader(img_datasets[key], batch_size=configs['batch_size'], shuffle=True)
                       for key in img_datasets}

    # ----------------->>> loss and optimisers
    loss_fn = nn.CrossEntropyLoss()  # KLDivLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=configs['learning_rate'])

    # ----------------->>> starting the run
    run = wandb.init(project="Flower102", entity="3it_dot_classifier", id=ID, resume="allow",
                     config=configs, tags=['resnet50'])  # notes, tags, group  are other usfull parameters
    RUN_NAME = run.name
    basicTrainingSequence(model, loss_fn, optimizer, img_dataloaders['test'],
                          img_dataloaders['train'], configs['epochs'])


if __name__ == '__main__':
    main()


