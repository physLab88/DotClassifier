import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from torchvision import models
import cronos as cron
import wandb
import os

# ===================== GLOBAL VARIABLES =====================
RUN_NAME = ''
ID = wandb.util.generate_id()
NETWORK_DIRECTORY = "Networks/Flower102/"
PROJECT = 'Flower102'
ENTITY = "3it_dot_classifier"
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
            # print(X.size())
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


def basicTrainingSequence(model, loss_fn, optimizer, train_dataloader, test_dataloader, numEpoch, init_epoch=1):
    # TODO initial assestement
    # TODO continue run / checkpoint ???
    # ------------->>> training loop
    for E in range(numEpoch):
        print("Starting Epoch %s \n ========================================" % (E+1))
        train_loop(train_dataloader, model, loss_fn, optimizer)
        accu, loss = test_loop(test_dataloader, model, loss_fn)

        wandb.log({"loss": loss, "accuracy": accu, "epoch": E + init_epoch})
        if True:
            fig1, ax = plt.subplots()
            plt.plot(np.arange(5), np.arange(5)*E)
            # fig2, ax = plt.subplots()
            # plt.plot(np.arange(5), np.arange(5))
            wandb.log({"epoch": E + init_epoch, "fig1": fig1})
        wandb.config.update({"epochs": E + init_epoch}, allow_val_change=True)
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


def load_model(model_name, configs, branch_training=False, tags=None):
    ''' This function allows to load previous neural nets
    filepath: the name of the model (this model must be in the
              NETWORK_DIRECTORY directory)
    branch_training: if false, training or testing will resume on
                     that network. if trained, the new network will be
                     saved on top of this network. if True, will branch
                     out the network and create a new ID for the network'''
    # TODO checkpoint resume
    global ID

    # ---------------->>> Loading the model
    model_file_id = None
    for filename in os.listdir(NETWORK_DIRECTORY):
        f = os.path.join(NETWORK_DIRECTORY, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if filename.find(model_name) != -1:
                model_file_id = filename[0:filename.find('.')]
                break
    if model_file_id is None:
        raise 'model_name= %s not found in %s directory' % (model_name, NETWORK_DIRECTORY)
    model = torch.jit.load(NETWORK_DIRECTORY + model_file_id + '.pt')
    model.train()
    model = model.to(device)

    api = wandb.Api()
    run_id = model_file_id[model_file_id.find('_') + 1:]
    temp_run = api.run("{entity}/{project}/{run_id}".format(entity=ENTITY, project=PROJECT, run_id=run_id))
    prev_config = temp_run.config
    summary = temp_run.summary

    configs['architecture'] = prev_config['architecture']
    configs['pretrained'] = prev_config['pretrained']
    init_epoch = summary['epoch']
    if not branch_training:
        ID = model_file_id[model_file_id.find('_') + 1:]
        print('ID %s' % ID)
        configs['parent_run'] = model_name

    run = wandb.init(project=PROJECT, entity=ENTITY, id=ID, resume="allow",
                     config=configs, tags=tags)  # notes, tags, group  are other usfull parameters

    if branch_training:
        wandb.log({"loss": summary['loss'], "accuracy": summary['accuracy'], "epoch": init_epoch})
    return run, model, init_epoch + 1


# ============================ MAIN ============================
def main():
    # TODO delete epochs config
    # TODO randomise weights
    # TODO implement parent run everywhere
    global ID, RUN_NAME
    configs = {
        "learning_rate": 1E-3,
        "epochs": 5,
        "batch_size": 16,
        "architecture": "ResNet50",     # modified when loaded
        "pretrained": False,            # modified when loaded
        "loss_fn": "CrossEntropyLoss",
        "optimiser": "Adam",
    }
    tags = ['resnet50']
    img_dataloaders = {key: DataLoader(img_datasets[key], batch_size=configs['batch_size'], shuffle=True)
                       for key in img_datasets}

    # ======================= BUILDING MODEL AND WANDB =======================
    model_name = None
    branch_training = True
    if model_name is None:
        model = models.resnet50(weights=False)
        # print(model._modules.keys())
        # print(model._modules['fc'])
        last_layer = 'fc'
        temp_in = model._modules[last_layer].in_features
        temp_out = 102
        model._modules[last_layer] = nn.Linear(temp_in, temp_out)
        model = model.to(device)

        run = wandb.init(project=PROJECT, entity=ENTITY, id=ID, resume="allow",
                         config=configs, tags=tags)  # notes, tags, group  are other usfull parameters
        init_epoch = 1
    else:
        run, model, init_epoch = load_model(model_name, configs,  branch_training=branch_training, tags=tags)

    # ----------------->>> loss and optimisers
    loss_fn = nn.CrossEntropyLoss()  # KLDivLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'])

    # ----------------->>> training the network
    RUN_NAME = run.name
    basicTrainingSequence(model, loss_fn, optimizer, img_dataloaders['test'],
                          img_dataloaders['train'], configs['epochs'], init_epoch=init_epoch)


if __name__ == '__main__':
    main()


