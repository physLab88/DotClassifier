import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from math import ceil

from torchvision import models
import yaml
import cronos as cron
import wandb
import os

# ===================== GLOBAL VARIABLES =====================
RUN_NAME = ''
ID = wandb.util.generate_id()
NETWORK_DIRECTORY = "Networks/LazyEcMesure/"
PROJECT = 'LazyEcMesure'
ENTITY = "3it_dot_classifier"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
DIRECTORY = "data/sim2_0/"
BATCH_SIZE = None
RANDOM_CROP = True


# ======================= SETTING UP DATASET ========================
# ---------------->>> DATALOADER
class StabilityDataset(Dataset):
    """ Class that allows to retreive the training and validation data"""
    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        f = open(root_dir + '_data_indexer.yaml', 'r')
        self.info = yaml.load(f, Loader=yaml.FullLoader)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        return

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_name = self.root_dir + self.info[idx]['f'] + '.npy'
        sample = np.load(sample_name)
        sample = np.float32(sample)

        target = torch.FloatTensor([self.info[idx]['Ec']])
        target_info = self.info[idx]

        # random crop:
        newBox = []
        if RANDOM_CROP:
            box = target_info['box']
            temp = randint(0, box[1][1] - 4)
            newBox = [[randint(box[0][0] + 4, target_info['nVg']), target_info['nVds'] - temp],
                      [randint(0, box[1][0]), temp],  # uper left corner
                      ]  # lower right
            MIN_SIZE = 33
            temp = newBox[0][0] - newBox[1][0]
            if temp < MIN_SIZE:
                temp = ceil((MIN_SIZE - temp)/2)
                newBox[0][0] += temp
                newBox[1][0] -= temp
                if newBox[1][0] < 0:
                    newBox[0][0] -= newBox[1][0]
                    newBox[1][0] = 0

            temp = newBox[0][1] - newBox[1][1]
            if temp < MIN_SIZE:
                temp = ceil((MIN_SIZE - temp)/2)
                newBox[1][1] -= temp
                newBox[0][1] += temp
            sample = sample[newBox[1][1]:newBox[0][1], newBox[1][0]:newBox[0][0]]
            # print("old: %s,\t new: %s" % (box, newBox))
        newBox = torch.IntTensor(newBox)

        # target['target'] = torch.FloatTensor([self.info[idx]['Ec']])
        sample = np.repeat(sample[:, :, None], 3, axis=2)  # to use with resnet50
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        # print(sample.size())
        return sample, target, idx  #, newBox


# a simple custom collate function, just to show the idea
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    idx = [item[2] for item in batch]
    # target = torch.LongTensor(target)
    return [data, target, idx]


class RandomMultiply(object):
    def __init__(self, min, max):
        self.min = min
        self.width = max - min

    def __call__(self, sample):
        return sample * (self.min + np.random.uniform()*self.width)


data_transforms = {'train': transforms.Compose([
    RandomMultiply(0.7, 1.3),
    transforms.ToTensor(),
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.025)),
    ]),
                   'valid': transforms.Compose([
    RandomMultiply(0.7, 1.3),
    transforms.ToTensor(),
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.025)),
    ]),
}
# transforms.GaussianBlur(kernel_size[, sigma])
data_target_transforms = {'train': None,
                          'valid': None,
                          }

folders = {'train': 'train/',
           'valid': 'valid/',}
img_datasets = {
    key: StabilityDataset(root_dir=DIRECTORY + folders[key], transform=data_transforms[key],
                          target_transform=data_target_transforms[key]) for key in data_transforms}


def lookAtData(dataloader, info, nrows=1, ncols=1):
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    info = info[0]
    Vg = info['Vg_range']
    Vds = info['Vds_range']
    for i in range(nrows):
        axs[i, 0].set_ylabel(r'$V_{ds}$ in mV')
        for j in range(ncols):
            plt.sca(axs[i, j])
            diagrams, labels, idx = next(iter(dataloader))
            #print(diagrams[0].size())
            index = np.random.randint(0, BATCH_SIZE)
            plt.title('Ec: %s meV' % '{:2f}'.format(float(labels[index])))
            plt.imshow(np.abs(diagrams[index][0]), extent=[Vg[0], Vg[-1], Vds[0], Vds[-1]], aspect=1, cmap='hot')
            plt.xlim([0, 290])
    for j in range(ncols):
        axs[-1, j].set_xlabel(r'$V_g$ in mV')
    #plt.tight_layout()
    plt.show()


# ==================== TRAINING AND TESTING ====================
def test_loop(dataloader, model, loss_fn):
    print("=========== Evaluating model ===========")
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        i = 0
        for X, y, index in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            i += 1
            #print('v %s' % i)
    # TODO implement std?
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    i=0
    cron.start('train_print')
    for batch, (X, y, index) in enumerate(dataloader):
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
    best = None
    best_summary = {}
    # ------------->>> training loop
    for E in range(numEpoch):
        print("Starting Epoch %s \n ========================================" % (E+1))
        train_loop(train_dataloader, model, loss_fn, optimizer)
        accu, loss = test_loop(test_dataloader, model, loss_fn)

        wandb.log({"loss": loss, "accuracy": accu, "epoch": E + init_epoch})
        wandb.config.update({"epochs": E + init_epoch}, allow_val_change=True)
        # ------------>>> make a checkpoint
        # ---> save report
        # ---> save optim
        # ---> save best model
        if best is None or loss <= best:
            best = loss
            best_summary["epoch"] = E + init_epoch
            best_summary["loss"] = loss
            best_summary["accuracy"] = accu

            model_scripted = torch.jit.script(model)  # Export to TorchScript
            model_scripted.save(NETWORK_DIRECTORY + RUN_NAME + '_' + ID + '.pt')  # Save
        # update the summary with the current best values
        for key in best_summary:
            wandb.run.summary[key] = best_summary[key]
    # ------------>>> final save
    # ---> save report
    # ---> save model
    # model_scripted = torch.jit.script(model)  # Export to TorchScript
    # model_scripted.save(NETWORK_DIRECTORY + RUN_NAME + '_' + ID + '.pt')  # Save


def load_model(model_name, configs=None, branch_training=True, tags=None, train=True):
    ''' This function allows to load previous neural nets
    filepath: the name of the model (this model must be in the
              NETWORK_DIRECTORY directory)
    configs: the current training run configurations. some of these
             configs will be automatically updated to match the previous runs.
             if no configs are given, the model will automatically be loaded
             with all the previous configs
    branch_training: if false, training or testing will resume on
                     that network. if trained, the new network will be
                     saved on top of this network. if True, will branch
                     out the network and create a new ID for the network
    train: wether or not load the model in train mode or eval mode'''
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
    if train:
        model.train()
    else:
        model.eval()
    model = model.to(device)

    api = wandb.Api()
    run_id = model_file_id[model_file_id.find('_') + 1:]
    temp_run = api.run("{entity}/{project}/{run_id}".format(entity=ENTITY, project=PROJECT, run_id=run_id))
    prev_config = temp_run.config
    summary = temp_run.summary

    if configs is None:
        configs = prev_config
    else:
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
    return run, model, init_epoch + 1, configs


# =================== CREATING A CUSTOM MODEL ==================
# not used for now
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class EndBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(EndBlock, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            # nn.Linear(input_size, input_size),
            # nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
        )
        self.out_layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear_relu_stack(x)
        logits = self.out_layer(out + x)  # skip connection
        return logits


# ======================== MAIN ROUTINES ========================
def analise_network(model_name, datatype='valid'):
    img_dataloaders = {key: DataLoader(img_datasets[key], batch_size=1, shuffle=True)
                       for key in img_datasets}
    dataloader = img_dataloaders[datatype]
    infos = img_datasets[datatype].info
    # lookAtData(dataloader, dataloader.info, 5, 5)
    loss_fn = nn.MSELoss()
    run, model, epoch, configs = load_model(model_name, branch_training=False, train=True)


    print("=========== Analising model fit ===========")
    loss = []
    error = []
    alpha = []
    Ec = []
    n_levels = []
    T = []
    with torch.no_grad():
        for X, y, index in dataloader:
            info = infos[index]
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            error.append(abs(float(pred - info['Ec'])))
            loss.append(loss_fn(pred, y).item())
            alpha.append(info['ag'])
            Ec.append(info['Ec'])
            n_levels.append(len(info['levels']))
            T.append(info['T'])
            if loss[-1] > 999999:
                X = X.to('cpu')
                X = X.numpy()[0, 0]
                plt.imshow(np.abs(X), aspect=1, cmap='hot')
                plt.show()
    loss = np.array(loss)
    error = np.array(error)
    alpha = np.array(alpha)
    Ec = np.array(Ec)
    n_levels = np.array(n_levels)

    figAlpha, ax = plt.subplots()
    plt.title('epoch = %s' % epoch)
    plt.plot(error, alpha, 'ok', ms=1)
    plt.xlabel("Ec error in meV")
    plt.ylabel("alpha")

    figEc, ax = plt.subplots()
    plt.title('epoch = %s' % epoch)
    plt.plot(error, Ec, 'ok', ms=1)
    plt.xlabel("Ec error in meV")
    plt.ylabel("Ec (meV)")

    figLv, ax = plt.subplots()
    plt.title('epoch = %s' % epoch)
    plt.plot(error, n_levels, 'ok', ms=1)
    plt.xlabel("Ec error in meV")
    plt.ylabel("number of energy levels")

    figT, ax = plt.subplots()
    plt.title('epoch = %s' % epoch)
    plt.plot(error, T, 'ok', ms=1)
    plt.xlabel("Ec error in meV")
    plt.ylabel("Temperature (K)")

    wandb.log({"alpha fig": figAlpha,
               "Ec fig": figEc,
               "n level fig": figLv,
               "T fig": figT,})
    return loss, alpha


def train():
    # TODO delete epochs config
    # TODO randomise weights
    # TODO implement parent run everywhere

    global ID, RUN_NAME, BATCH_SIZE
    BATCH_SIZE = 1
    configs = {
        "learning_rate": 1E-3,
        "epochs": 20,
        "batch_size": BATCH_SIZE,
        "architecture": "ResNet50",  # modified when loaded
        "pretrained": True,  # modified when loaded
        "loss_fn": "mean squared error loss",
        "optimiser": "SGD",
        "data_used": "first 2.0 random_crop",
        "data_size": len(img_datasets['train']),
        "valid_size": len(img_datasets['valid'])
    }
    tags = ['resnet50']
    print('Dataset train size = %s' % len(img_datasets['train']))
    img_dataloaders = {key: DataLoader(img_datasets[key], batch_size=BATCH_SIZE, shuffle=True)  # , collate_fn=my_collate)
                       for key in img_datasets}
    # lookAtData(img_dataloaders['train'], img_datasets['train'].info, 5, 5)

    # ======================= BUILDING MODEL AND WANDB =======================
    model_name = "playful-pine"
    branch_training = True  # always True unless continuing a checkpoint
    if model_name is None:
        model = models.resnet50(weights=True)
        # print(model._modules.keys())
        # print(model._modules['fc'])
        last_layer = 'fc'
        temp_in = model._modules[last_layer].in_features
        temp_out = 1
        # model._modules[last_layer] = EndBlock(temp_in, temp_out)
        model._modules[last_layer] = nn.Linear(temp_in, temp_out)
        model = model.to(device)

        run = wandb.init(project=PROJECT, entity=ENTITY, id=ID, resume="allow",
                         config=configs, tags=tags)  # notes, tags, group are other usfull parameters
        init_epoch = 1
    else:
        run, model, init_epoch, configs = load_model(model_name, configs, branch_training=branch_training, tags=tags)

    # ----------------->>> loss and optimisers
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=configs['learning_rate'])

    # ----------------->>> training the network
    RUN_NAME = run.name
    basicTrainingSequence(model, loss_fn, optimizer, img_dataloaders['train'],
                          img_dataloaders['valid'], configs['epochs'], init_epoch=init_epoch)


# ============================ MAIN ============================
def main():
    # img_dataloaders = {key: DataLoader(img_datasets[key], batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)
    #                    for key in img_datasets}
    # lookAtData(img_dataloaders['train'], img_datasets['train'].info, 5, 5)
    #train()
    analise_network("iconic-capybara", 'valid')


if __name__ == '__main__':
    main()


