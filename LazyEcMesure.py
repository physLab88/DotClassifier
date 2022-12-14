""" Author: Michael Bedard
This code is used to train a neural network for the 20%-80% approach
It has its own WandB project, and its own Networks file.
this program provides a way to easily add data augmentation, to view
the training dataset and experimental dataset, it also provides functions
to train neural networks while automatically loging usfull data into WandB.
it also provides usfull analisis functions to analise datasets and to analise
network convergence on anny dataloader."""
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from math import ceil, floor
from scipy.stats import beta

from torchvision import models
import yaml
import cronos as cron
import wandb
import os

import diagram_imperfections as di

# ===================== GLOBAL VARIABLES =====================
# these are globa variables used in other parts of the program
RUN_NAME = ''
ID = wandb.util.generate_id()
NETWORK_DIRECTORY = "Networks/LazyEcMesure/"
PROJECT = 'LazyEcMesure'  # WandB project folder
ENTITY = "3it_dot_classifier"  # WandB entity
device = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if you can
print(f"Using {device} device")
DIRECTORY = "data/sim3_0/"  # simulated data directory
EXP_DIRECTORY = "data/exp_box/"  # exp data directorry
BATCH_SIZE = 16
DROPOUT = 0.30  # used if using dropout layers
NUM_GROUPS = 4  # used if using groupe norm

# needs to be defined here for uses later
exp_dataloader = None
sim_dataloaders = None


# ======================= SETTING UP DATASET ========================
# ---------------->>> DATALOADER CLASSES
class StabilityDataset(Dataset):
    """ dataset class that allows to retreive simulated data"""

    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        root_dir (string): Directory of the dataset.
        transform (callable, optional): Optional transform to be applied
                on a sample.
                target_transform: Optional transform to be aplied on target
        """
        f = open(root_dir + '_data_indexer.yaml', 'r')
        self.info = yaml.load(f, Loader=yaml.FullLoader)  # loading the dic with the targets and other info of samples
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        return

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        """ returns the sample and target at index idx"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # ----------------->>> loading the sample
        sample_name = self.root_dir + self.info[idx]['f'] + '.npy'
        sample = np.load(sample_name)
        sample = np.float32(sample)
        target_info = self.info[idx]
        target = torch.FloatTensor([self.info[idx]['Ec']])
        # here, we define the target Ec in Vds_pixels
        target = target / ((target_info["Vds_range"][1] - target_info["Vds_range"][0]) / (target_info["nVds"] - 1))
        # target /= (target_info["nVds"] - 1)  # if you want to normalise to image height

        # ----------------->>> data augmentation
        # ---> threshold current
        # sample = di.threshold_current(sample, target_info)
        # ---> 3D map
        # sample = np.log(np.abs(di.clip_current(sample, 2E-14)))
        # sample = di.low_freq_3DMAP(sample, target_info)
        # sample = np.exp(sample)
        # ---> gaussian blur
        sample = di.gaussian_blur(sample, target_info, 0, 3.0)  # 1.0, 5.0)
        # ---> random current modulation
        # sample = di.rand_current_addition(sample, target_info, beta.rvs(0.8, 4, 1E-3, 0.07))
        # sample = di.rand_current_modulation(sample, target_info, 0.5)
        # ---> white noise
        sample = di.white_noise(sample, np.exp(beta.rvs(0.85, 1.5, -30.5, 2)))
        # ---> crop
        # sample, newBox = di.random_crop(sample, target_info)  # random crop
        sample, newBox = di.diamond_crop(sample, target_info)  # or diamond crop
        # ---> scaling
        sample = di.random_multiply(sample, np.exp(beta.rvs(2.8, 3.7, -28, 8.5)))

        # ----------------->>> formatting the data
        sample, mask = di.black_square(sample)
        del mask
        sample = di.clip_current(sample, 2E-14, 1E-7)
        sample = np.log(np.abs(sample))
        # sample = np.repeat(sample[:, :, None], 3, axis=2)  # to use with resnet50
        sample = sample.astype('float32')

        # ----------------->>>  other transforms
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        # print(sample.size())
        return sample, target, idx  # we return the idx for analisis


class ExperimentalDataset(Dataset):
    """ dataset class that allows to retreive the experimental data"""

    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        root_dir (string): Directory of the dataset.
        transform (callable, optional): Optional transform to be applied
                on a sample.
        target_transform: Optional transform to be aplied on target
        """
        f = open(root_dir + '_data_indexer.yaml', 'r')
        self.info = yaml.load(f, Loader=yaml.FullLoader)  # loading the dic with the targets and other info of samples
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        return

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        """ returns the sample and target at index idx"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # ----------------->>> loading the sample
        sample_name = self.root_dir + self.info[idx]['f'] + '.npy'
        sample = np.load(sample_name)
        sample = np.float32(sample)
        target_info = self.info[idx]
        target = torch.FloatTensor([self.info[idx]['Ec']])
        # --- here, we define the target Ec in Vds_pixels
        target = target / ((target_info["Vds_range"][1] - target_info["Vds_range"][0]) / (target_info["nVds"] - 1))
        # target /= (target_info["nVds"] - 1)  # if you want to normalise to image height

        # ----------------->>> data augmentation
        # ---> crop
        # sample, newBox = di.random_crop(sample, target_info)  # random crop
        sample, newBox = di.diamond_crop(sample, target_info)  # or diamond crop
        # ---> resolution change
        # sample, target = di.change_res(sample, target)
        # ---> random scale
        # sample = di.random_multiply(sample, 0.5, 1.75)  # random scale

        # ----------------->>> formating the data
        sample, mask = di.black_square(sample)
        del mask
        sample = di.clip_current(sample, 2E-14, 1E-7)
        sample = np.log(np.abs(sample))
        # sample = np.repeat(sample[:, :, None], 3, axis=2)  # to use with resnet50
        sample = sample.astype('float32')

        # ----------------->>>  other transforms
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        # print(sample.size())
        return sample, target, idx  # we return the idx for analisis


# ---------------->>> defining other transforms
data_transforms = {'train': transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomAffine(degrees=0, translate=(0, 0.025)),  # random translation
]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.025)),  # random translation
    ]),
}
data_target_transforms = {'train': None,
                          'valid': None,
                          }

exp_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomVerticalFlip(),])

folders = {'train': 'train/',
           'valid': 'valid/', }
# ----------->>> defining the datasets
sim_datasets = {
    key: StabilityDataset(root_dir=DIRECTORY + folders[key], transform=data_transforms[key],
                          target_transform=data_target_transforms[key]) for key in data_transforms}
exp_dataset = ExperimentalDataset(root_dir=EXP_DIRECTORY, transform=exp_transform)


# ============================== TOOLS ==============================
def lookAtData(dataloader, info, nrows=2, ncols=2):
    """ This function allows you to look at manny data at a time.
    it is meant to be modified if you need to.
    dataloader: the dataloader from which you wish to look at
    info: the self.info dic of the dataloader
    nrows: the number of image rows in the output graph
    ncols: the number of image columns in the graph"""
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    info = info[0]
    Vg = info['Vg_range']
    Vds = info['Vds_range']
    for i in range(nrows):
        axs[i, 0].set_ylabel(r'$V_{ds}$ in mV')
        for j in range(ncols):
            plt.sca(axs[i, j])

            # loading a random file (does not work if BATCH_SIZE = 1, I think
            diagrams, labels, idx = next(iter(dataloader))
            index = np.random.randint(0, BATCH_SIZE)
            diagram = diagrams[index][0]
            size = diagram.size()

            plt.title('h:%spx' % '{:.2f}'.format(float(labels[index])))
            plt.imshow(diagram, aspect=1, cmap='hot')  # extent=[Vg[0], Vg[-1], Vds[0], Vds[-1]]
    for j in range(ncols):
        axs[-1, j].set_xlabel(r'$V_g$ in mV')
    # plt.tight_layout()
    plt.show()


def look_at_exp():
    """ This function allows to visualise each experimental data
    one diagram at a time. note that is actually creates a new exp
    dataloader from the global exp_dataset and does not use the global
    exp_dataloader"""
    dataloader = DataLoader(exp_dataset, shuffle=False)
    infos = exp_dataset.info
    for diagram, label, index in dataloader:
        diagram = diagram[0, 0]
        label = label[0]
        info = infos[index[0]]
        Vg = info['Vg_range']
        Vds = info['Vds_range']
        plt.title('Ec: %s meV  pixel height %s' % ('{:2f}'.format(float(info['Ec'])), '{:2f}'.format(float(label))))
        plt.imshow(diagram, aspect=1, cmap='hot')  # , extent=[Vg[0], Vg[-1], Vds[0], Vds[-1]])
        plt.show()


def get_input_stats(dataloader, multiplot=False, title=''):
    """this function analises the statistics of anny dataloader.
    it is meant to be modified"""
    means = []
    stds = []
    for i in range(3):
        for X, y, index in dataloader:
            means.append(X.mean())
            stds.append(X.std())
    plt.hist(stds)
    plt.title(title)
    if not multiplot:
        plt.show()
    return


# ============= TRAIN/TEST BACKGROUND FUNCTIONS =============
def test_loop(dataloader, model, loss_fn):
    """ This function loops 3 times over the dataloader to mesure the loss.
    it is usefull when you want to log your fit over your validation data
    and over your experimental data
    dataloader: anny dataloader you want to test your model on
    model: your trained model
    loss_fn: the loss function to use during the test"""
    print("=========== Evaluating model ===========")
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        i = 0
        for k in range(3):
            for X, y, index in dataloader:
                # ---> running the model
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                i += 1
    test_loss /= size * 3
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss


def train_loop(dataloader, model, loss_fn, optimizer):
    """ This function loops over the dataloader and trains the model
    exactly 1 epoch.
    dataloader: the training data to use
    model: the model to train
    loss_fn: the loss function to minimise
    optimiser: the algorythm that will optimise the model"""
    size = len(dataloader.dataset)
    i = 0
    cron.start('train_print')
    running_loss = 0
    for batch, (X, y, index) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # ---> Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        running_loss += loss.item()

        # ---> Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1

        # ---> printing timer
        if cron.t('train_print') >= 5.0:
            cron.start('train_print')
            current = (batch + 1) * len(X)
            temp = current / size
            bar = '[' + int(temp * 30) * '#' + int((1 - temp) * 30) * '-' + ']'
            print("loss: {loss:.5f}  {bar:>} {perc:.2%}".format(loss=running_loss / current, bar=bar, perc=temp))
    cron.stop('train_print')
    return running_loss / size


def basicTrainingSequence(model, loss_fn, optimizer, train_dataloader, test_dataloader, numEpoch, init_epoch=1,
                          exp_dataloader=None):
    """ this function takes care of evrything that needs to be done during training:
    training, testing, logging to WandB, saving the model and etc.
    it will train the model numEpoch epoch.

    model: the model to train
    loss_fn: the loss function to minimise
    train_dataloader: the simulated dataloader to use during training
    test_dataloader: the simulated dataloader to use during validation
    exp_dataloader: the dataloader to use during validation, to verify if exp converges
        if None, skips the exp data validation
    numEpoch: the number of epoch to train for
    init_epoch: the model's starting epoch (used when loading a previously trained model)
    """
    # TODO continue run / checkpoint
    best = None
    best_summary = {}
    # ------------->>> training loop
    for E in range(numEpoch):
        print("Starting Epoch %s \n ========================================" % (E + 1))
        # ---> training a new epoch
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        loss = test_loop(test_dataloader, model, loss_fn)

        # ---> loging points to WandB
        log_dic = {"loss": loss, "train loss": train_loss, "epoch": E + init_epoch}
        exp_loss = None
        if exp_dataloader is not None:
            exp_loss = test_loop(exp_dataloader, model, loss_fn)
            log_dic["exp loss"] = exp_loss
        wandb.log(log_dic)
        wandb.config.update({"epochs": E + init_epoch}, allow_val_change=True)

        # ---> verify if we got a new best fit
        if best is None or loss <= best:
            # ---> update the summary with the current best values
            best = loss
            best_summary["epoch"] = E + init_epoch
            best_summary["loss"] = loss
            best_summary["train loss"] = train_loss
            if exp_dataloader is not None:
                best_summary["exp loss"] = exp_loss
            for key in best_summary:
                wandb.run.summary[key] = best_summary[key]

            # ---> save the new best model
            model_scripted = torch.jit.script(model)  # Export to TorchScript
            model_scripted.save(NETWORK_DIRECTORY + RUN_NAME + '_' + ID + '.pt')  # Save
    return


def load_model(model_name, configs=None, branch_training=True, tags=None, train=True):
    ''' This function allows to load previously saved neural nets
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
    # ---> finding the model's file
    for filename in os.listdir(NETWORK_DIRECTORY):
        f = os.path.join(NETWORK_DIRECTORY, filename)
        if os.path.isfile(f):
            if filename.find(model_name) != -1:
                model_file_id = filename[0:filename.find('.')]
                break
    if model_file_id is None:
        raise 'model_name= %s not found in %s directory' % (model_name, NETWORK_DIRECTORY)
    # ---> actually loading the model
    model = torch.jit.load(NETWORK_DIRECTORY + model_file_id + '.pt')
    if train:
        model.train()
    else:
        model.eval()
    model = model.to(device)

    # -------------->>> updating some configs using the previous configs
    api = wandb.Api()
    run_id = model_file_id[model_file_id.find('_') + 1:]
    temp_run = api.run("{entity}/{project}/{run_id}".format(entity=ENTITY, project=PROJECT, run_id=run_id))
    prev_config = temp_run.config
    summary = temp_run.summary

    init_epoch = summary['epoch']
    if configs is None:
        configs = prev_config
    else:
        configs['architecture'] = prev_config['architecture']
        configs['pretrained'] = prev_config['pretrained']

    # ---> copying the previous ID if don't branch training
    if not branch_training:
        ID = model_file_id[model_file_id.find('_') + 1:]
        print('ID %s' % ID)
    else:
        configs['parent_run'] = model_name

    # ---> Creating a new WandB run
    run = wandb.init(project=PROJECT, entity=ENTITY, id=ID, resume="allow",
                     config=configs, tags=tags)  # notes, tags, group  are other usfull parameters

    if branch_training:
        wandb.log({"loss": summary['loss'], "accuracy": summary['accuracy'], "epoch": init_epoch})
    return run, model, init_epoch + 1, configs


# ========================= CREATING A CUSTOM MODEL =========================
class Bottleneck(nn.Module):
    """ Defining one of the principal building blocks of the Reslike1_0 net
    it is inspired by the Resnet50 Architecture"""
    def __init__(self, in_planes, mid_planes, out_planes, bias=True, stride=1):
        super(Bottleneck, self).__init__()
        # bias is usaly False in resnet
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, bias=bias)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, bias=bias)
        self.batchNormLike = nn.InstanceNorm2d(out_planes)  # a layer that replaces the batchnorm layer
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_planes != out_planes or stride != 1:
            self.downsample = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)

        pass

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchNormLike(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batchNormLike(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.batchNormLike(out)

        # skip connection
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        out = self.relu(out)
        return out

    pass


class ResLike1_0(nn.Module):
    """This network is inspired from the resnet50 Architecture. it was built
    to modify the architecture and replace the batchnorm layers"""
    def __init__(self):
        super(ResLike1_0, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.relu = nn.ReLU(inplace=True)
        # should technically be used after conv1, but I didn't use it and it works fine
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.sequ1 = nn.Sequential(
            Bottleneck(64, 64, 256),
            Bottleneck(256, 64, 256),
            Bottleneck(256, 64, 256),
        )
        self.sequ2 = nn.Sequential(
            Bottleneck(256, 128, 512, stride=2),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512),
        )
        self.sequ3 = nn.Sequential(
            Bottleneck(512, 256, 1024, stride=2),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
        )
        self.sequ4 = nn.Sequential(
            Bottleneck(1024, 512, 2048, stride=2),
            Bottleneck(2048, 512, 2048),
            Bottleneck(2048, 512, 2048),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.dropout = nn.Dropout(DROPOUT, inplace=True)  # if you want to put dropout
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.sequ1(x)
        x = self.sequ2(x)
        x = self.sequ3(x)
        x = self.sequ4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        # x = self.dropout(x)  # only put dropout at end as dropout not recomended in conv layers
        x = self.lin1(x)
        return x


class EndBlock(nn.Module):
    """ This was used to modify the last layer of the resnet50 architecture
    for experimentation purposes"""
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


def sets_running_stats(node):
    """ This function is used to set the 'running_stats' of every BatchNorm layers 
    of the resnet50 architecture to False"""
    try:
        if type(node) == nn.BatchNorm2d:
            node.track_running_stats = False
            return
        for key in node._modules.keys():
            sets_running_stats(node._modules[key])
    except:
        if type(node) == nn.BatchNorm2d:
            node.track_running_stats = False


# ======================== MAIN ROUTINES ========================
def analise_network(model_name, datatype='valid'):
    """ This function is extreamly usfull to analise your model fitting after
    training, as this function plots manny graphs of the error as respect to
    other parameters such as diamond height resolution, diamond width resolution
    and etc, to verify that the fitting went well and see if there are anny
    dependance between the error and other parameters.
    This function Logs the results in WandB

    this is however only used for simulated data files
    Note!!!: the data must be using batch of 1"""
    dataloader = sim_dataloaders[datatype]
    infos = sim_datasets[datatype].info
    loss_fn = nn.MSELoss()
    run, model, epoch, configs = load_model(model_name, branch_training=False, train=True)

    print("=========== Analising model fit ===========")
    loss = []
    error = []
    alpha = []
    Ec = []
    n_levels = []
    T = []
    Vds_res = []
    Vg_res = []
    square_res = []
    for k in range(3):
        with torch.no_grad():
            for X, y, index in dataloader:
                # ---> running the model
                info = infos[index]
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                # ---> calculating and loging the metrics

                error.append(abs(float((pred - y) / y)) * 100.0)  # error in %
                loss.append(loss_fn(pred, y).item())
                alpha.append(info['ag'])
                Ec.append(info['Ec'])
                n_levels.append(len(info['levels']))
                T.append(info['T'])
                Vds_res.append(float(y))
                Vg_res.append(float(y / info['ag']))
                square_res.append(float(y ** 2 + (y / info['ag']) ** 2))
                if error[-1] > 999999:
                    """ Use this if statement to instantly visualise data that
                    has huge errors"""
                    X = X.to('cpu')
                    X = X.numpy()[0, 0]
                    plt.imshow(X, aspect=1, cmap='hot')
                    plt.show()
    loss = np.array(loss)
    error = np.array(error)
    alpha = np.array(alpha)
    Ec = np.array(Ec)
    n_levels = np.array(n_levels)
    Vg_res = np.array(Vg_res)
    Vds_res = np.array(Vds_res)
    square_res = np.array(square_res)

    # ---------->>> Creating every graph
    figAlpha, ax = plt.subplots()
    plt.title('epoch = %s' % epoch)
    plt.plot(error, alpha, 'ok', ms=1)
    plt.xlabel("relative Ec error in %")
    plt.ylabel("alpha")

    figEc, ax = plt.subplots()
    plt.title('epoch = %s' % epoch)
    plt.plot(error, Ec, 'ok', ms=1)
    plt.xlabel("relative Ec error in %")
    plt.ylabel("Ec (meV)")

    figLv, ax = plt.subplots()
    plt.title('epoch = %s' % epoch)
    plt.plot(error, n_levels, 'ok', ms=1)
    plt.xlabel("relative Ec error in %")
    plt.ylabel("number of energy levels")

    figT, ax = plt.subplots()
    plt.title('epoch = %s' % epoch)
    plt.plot(error, T, 'ok', ms=1)
    plt.xlabel("relative Ec error in %")
    plt.ylabel("Temperature (K)")

    figVds_res, ax = plt.subplots()
    plt.title('epoch = %s' % epoch)
    plt.plot(error, Vds_res, 'ok', ms=1)
    plt.xlabel("relative Ec error in %")
    plt.ylabel("Vds number of pixel per height")

    figVg_res, ax = plt.subplots()
    plt.title('epoch = %s' % epoch)
    plt.plot(error, Vg_res, 'ok', ms=1)
    plt.xlabel("relative Ec error in %")
    plt.ylabel("Vg number of pixel per width")

    figSquare_res, ax = plt.subplots()
    plt.title('epoch = %s' % epoch)
    plt.plot(error, square_res, 'ok', ms=1)
    plt.xlabel("relative Ec error in %")
    plt.ylabel("square sum of pixels per diamonds")

    # ------->>> loging every graph to WandB
    wandb.log({"alpha fig": figAlpha,
               "Ec fig": figEc,
               "n level fig": figLv,
               "T fig": figT,
               "Vds_res_fig": figVds_res,
               "Vg_res_fig": figVg_res,
               "square_res_fig": figSquare_res, })
    return loss, alpha


def test_on_exp(model_name):
    """ This function is extreamly usfull to analise your model fitting after
        training. It is equivalent to analise_network, but is designed for
        experimental data (as experimental data have less info in their info dic)
        This function Logs the results in WandB

        this is however only used for experimentel data files"""
    dataloader = DataLoader(exp_dataset, batch_size=1, shuffle=False)
    infos = exp_dataset.info
    # lookAtData(dataloader, dataloader.info, 5, 5)
    loss_fn = nn.MSELoss()
    run, model, epoch, configs = load_model(model_name, branch_training=False, train=True)
    error = []
    loss = []
    Ec_error = []
    Ec = []
    Ec_guess = []
    ys = []
    preds = []
    with torch.no_grad():
        for X, y, index in dataloader:
            # ---> running the model
            info = infos[index]
            X = X.to(device)
            y = y.to(device)
            pred = model(X)

            # ---> calculating and loging the metrics
            ys.append(float(y))
            preds.append(float(pred))
            error.append(abs(float((pred - y) / y)) * 100.0)  # error in %
            loss.append(loss_fn(pred, y).item())
            Ec.append(info['Ec'])
            Ec_error.append(error[-1] * info['Ec'] / 100)
            temp = float(info['Ec'] * pred / y)
            Ec_guess.append(temp)
    loss = np.array(loss)
    error = np.array(error)
    Ec_error = np.array(Ec_error)
    Ec = np.array(Ec)
    Ec_guess = np.array(Ec_guess)

    # ---------->>> Creating every graph
    figExp_perc, ax = plt.subplots()
    plt.title('epoch = %s' % epoch)
    plt.plot(Ec, error, 'ok', ms=1)
    plt.xlabel("Ec value in mV")
    plt.ylabel("relative Ec error in %")

    figExp_abs, ax = plt.subplots()
    plt.title('epoch = %s' % epoch)
    plt.plot(Ec, Ec_error, 'ok', ms=1)
    plt.xlabel("Ec value in mV")
    plt.ylabel("Ec error in mV")

    figExp_guess, ax = plt.subplots()
    plt.title('epoch = %s' % epoch)
    plt.plot(Ec, Ec_guess, 'ok', ms=1)
    plt.xlabel("Ec value in mV")
    plt.ylabel("Ec guess in mV")

    figExp_raw_y, ax = plt.subplots()
    plt.title('epoch = %s' % epoch)
    plt.plot(ys, preds, 'ok', ms=1)
    plt.xlabel("wanted height in pixels")
    plt.ylabel("height prediction in pixels")

    # ------->>> loging every graph to WandB
    wandb.log({"exp_perc": figExp_perc,
               "exp_abs": figExp_abs,
               "exp_guess": figExp_guess,
               "exp_raw_y": figExp_raw_y})
    return


def train():
    """ Call this function in your main to train a network. it will run
    the training sequence based on the parameters you have set inside
    this function"""
    # TODO implement checkpoint (right now, you can only branch training, wich does the trick)
    global ID, RUN_NAME, BATCH_SIZE, sim_dataloaders, exp_dataloader
    # =========================== CONFIGURATIONS =================================
    # NOTE: some configs are in the GLOBAL section at the top of the code
    configs = {
        "learning_rate": 1E-3,
        "epochs": 30,                   # here, set the max #epoch you want to train for
        "batch_size": BATCH_SIZE,
        "architecture": "ResLike3_3",   # modified when loaded
        "pretrained": True,             # modified when loaded
        "loss_fn": "mean squared error loss",
        "optimiser": "Adam",
        "data_used": "3.0 di-crop",     # small description
        "data_size": len(sim_dataloaders['train'].dataset),  # train dataset size
        "valid_size": len(sim_dataloaders['valid'].dataset),
        "exp_data_size": len(exp_dataloader.dataset),
        "running_stats": False,
        "dropout": DROPOUT,
    }
    tags = ['ResLike2_0']
    print('Dataset train size = %s' % len(sim_datasets['train']))
    sim_dataloaders = {key: DataLoader(sim_datasets[key], batch_size=BATCH_SIZE, shuffle=True)
                       for key in sim_datasets}
    # No need to enter full model name, just the 2 words does the tric. ex: 'kind-dawn' or 'eager-smoke'
    model_name = None  # put the name of the model you want to load, None to start a new model

    # ======================= BUILDING MODEL AND WANDB =======================
    branch_training = True  # always True unless continuing a checkpoint
    if model_name is None:
        # ---> create a new model
        model = ResLike1_0()
        model = model.to(device)

        run = wandb.init(project=PROJECT, entity=ENTITY, id=ID, resume="allow",
                         config=configs, tags=tags, group='debug')  # notes, tags, group are other usfull parameters
        init_epoch = 1
    else:
        # ---> load a model
        run, model, init_epoch, configs = load_model(model_name, configs, branch_training=branch_training, tags=tags)

    # ----------------->>> loss and optimisers
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'])

    # ----------------->>> training the network
    RUN_NAME = run.name
    basicTrainingSequence(model, loss_fn, optimizer, sim_dataloaders['train'],
                          sim_dataloaders['valid'], configs['epochs'], init_epoch=init_epoch,
                          exp_dataloader=exp_dataloader)


# ================================= MAIN =================================
def main():
    # --------------->>> creating the dataloaders
    global exp_dataloader, sim_dataloaders
    exp_dataloader = DataLoader(exp_dataset, batch_size=1, shuffle=False)
    sim_dataloaders = {key: DataLoader(sim_datasets[key], batch_size=BATCH_SIZE, shuffle=True)
                       for key in sim_datasets}

    # --------------->>> functions to uncomment depending on what you want to do
    # get_input_stats(sim_dataloaders['valid'], title='valid mean')
    # get_input_stats(exp_dataloader,  title='exp mean')

    # for i in range(10):
    # lookAtData(sim_dataloaders['train'], sim_datasets['train'].info, 4, 8)
    # look_at_exp()
    train()
    # test_on_exp("wild-yogurt")
    # analise_network("wild-yogurt", 'valid')


if __name__ == '__main__':
    main()
