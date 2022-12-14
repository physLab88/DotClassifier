""" Author: Michael Bedard
This Code is to test a GAN approach. specifically, it tests the
wierd GAN approach I thought of and explained in the word document"""
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
import numpy as np
from numpy.random import randint, normal
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
RUN_NAME = ''
ID = wandb.util.generate_id()
GEN_NETWORK_DIRECTORY = "Networks/GAN_2_step/generator_nets/"
NETWORK_DIRECTORY = "Networks/GAN_2_step/regression_nets/"
PROJECT = 'GAN_2_step'
ENTITY = "3it_dot_classifier"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
DIRECTORY = "data/sim3_0/"          # simulated data directorry
EXP_DIRECTORY = "data/exp_box/"     # experimental data directorry
BATCH_SIZE = 32
# ------------->>> GAN parameters
RANDOM_VECT_SIZE = 100
BACK_FEED = 16                      # number of backfeed layers (in the network definition)
EXP_DATA_DEGENERANCY = 300          # number of times exp data is repeated in the disc dataloader
RESET_RATE = 7                      # number of epoch befor we reset the disc network
TRAIN_SEGMENTATION = 40             # number of batch to train the disc before training the generator

# -------------- >>> global dataloaders
exp_dataloader = None
sim_dataloaders = None
disc_dataloader = None
gan_dataloader = None
valid_dataloader = None
# -------------- >>> global models
discriminators = None
gan_model = None


# ======================= SETTING UP DATASET ========================
# ---------------->>> DATALOADER CLASSES
class StabilityDataset(Dataset):
    """ dataset class that allows to retreive simulated data"""

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
        # ---> crop
        # sample, newBox = di.random_crop(sample, target_info)  # random crop
        sample, newBox = di.diamond_crop(sample, target_info)  # or diamond crop
        # ---> scaling
        sample = di.random_multiply(sample, np.exp(beta.rvs(2.8, 3.7, -28, 8.5)))

        # ----------------->>> formatting the data
        sample, mask = di.black_square(sample)
        mask = torch.BoolTensor(mask)
        sample = di.clip_current(sample, 2E-14, 1E-7)
        sample = np.log(np.abs(sample))
        sample = sample.astype('float32')

        # ----------------->>>  other transforms
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        # ---> matching the mask # of dimentions to sample # of dimentions
        if len(sample.size()) == 3:
            mask = mask[None,]
        # print(sample.size())
        return sample, target, idx, mask  # we return the idx for analisis


class ExperimentalDataset(Dataset):
    """ dataset class that allows to retreive the experimental data"""

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
        # ---> crop
        sample, newBox = di.diamond_crop(sample, target_info)  # diamond crop
        # ---> resolution change
        # sample, target = di.change_res(sample, target)  # change resolution
        # ---> random scale
        sample = di.random_multiply(sample, 0.5, 1.75)  # random scale

        # ----------------->>> formating the data
        sample, mask = di.black_square(sample)
        mask = torch.BoolTensor(mask)
        sample = di.clip_current(sample, 2E-14, 1E-7)
        sample = np.log(np.abs(sample))
        sample = sample.astype('float32')

        # ----------------->>>  other transforms
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        # ---> matching the mask # of dimentions to sample # of dimentions
        if len(sample.size()) == 3:
            mask = mask[None,]
        return sample, target, idx, mask


class GenDataset(Dataset):
    """ This is a dataset that the generator model would generate. it can also be
    mixed with some experimental data. This dataloader uses the global 'gan_model'
    to generate new samples"""
    def __init__(self, sim_dataset, exp_dataset=None, transform=None, target_transform=None):
        """
        sim_dataset: a simulation dataset that our gen model will use to generate
            some new data
        exp_dataset: an experimental dataset if we want to mix experimental data with
            generated data. leave None if you want only generated data
        transform (callable, optional): Optional transform to be applied
                on a sample.
        target_transform: Optional transform to be aplied on target
        """
        self.sim_dataset = sim_dataset
        self.sim_len = len(sim_dataset)
        self.exp_dataset = exp_dataset
        self.exp_len = 0
        if exp_dataset is not None:
            self.exp_len = len(exp_dataset)
        self.transform = transform
        self.target_transform = target_transform
        return

    def __len__(self):
        return self.sim_len + EXP_DATA_DEGENERANCY*self.exp_len

    def __getitem__(self, idx):
        """ returns the sample and target at index idx"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # experimental data have the lower indexes, generated data have the larger indexes
        if idx < EXP_DATA_DEGENERANCY*self.exp_len:
            # ---> experimental data
            sample, target, idx, mask = self.exp_dataset[idx%self.exp_len]
            sample = sample.to(device)
            idx = -(idx + 1)  # in my convention, the returned index is negative if it represents an exp data
        else:
            # ---> generated data
            sample, target, idx, mask = self.sim_dataset[idx - EXP_DATA_DEGENERANCY*self.exp_len]
            sample = sample.to(device)

            batch = int(sample.size()[0])
            rand_vect = torch.normal(1, 0, size=[batch, RANDOM_VECT_SIZE], device=device)
            sample = gan_model(sample, rand_vect)  # generate the data
            sample[mask==False] = np.log(2E-14)    # put the black square back

        # ----------------->>>  other transforms
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
        return sample, target, idx, mask


# ---------------->>> defining other transforms
data_transforms = {'train': transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomAffine(degrees=0, translate=(0, 0.025)),
]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.025)),
    ]),
}
data_target_transforms = {'train': None,
                          'valid': None,
                          }

exp_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomVerticalFlip(),])


folders = {'train': 'train/',
           'valid': 'valid/', }
# ----------->>> defining the non generated datasets
sim_datasets = {
    key: StabilityDataset(root_dir=DIRECTORY + folders[key], transform=data_transforms[key],
                          target_transform=data_target_transforms[key]) for key in data_transforms}
exp_dataset = ExperimentalDataset(root_dir=EXP_DIRECTORY, transform=exp_transform)


# ============================== TOOLS ==============================
def lookAtData(dataloader, info, nrows=1, ncols=1, show=True):
    """ This function allows you to look at manny data at a time.
        it is meant to be modified if you need to.
        dataloader: the dataloader from which you wish to look at
        info: the self.info dic of the dataloader
        nrows: the number of image rows in the output graph
        ncols: the number of image columns in the graph
        show: wether or not to show the plot"""
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    info = info[0]
    Vg = info['Vg_range']
    Vds = info['Vds_range']
    for i in range(nrows):
        axs[i, 0].set_ylabel(r'$V_{ds}$ in mV')
        for j in range(ncols):
            plt.sca(axs[i, j])

            # loading a random file (does not work if BATCH_SIZE = 1, I think
            diagrams, labels, idx, mask = next(iter(dataloader))
            index = np.random.randint(0, BATCH_SIZE)
            diagram = diagrams[index][0]
            diagram = diagram.cpu().detach()  # data needs to be on cpu to plot
            size = diagram.size()

            plt.title('Pix_height: %s' % '{:2f}'.format(float(labels[index])))
            plt.imshow(diagram, aspect=1, cmap='hot')  # extent=[Vg[0], Vg[-1], Vds[0], Vds[-1]]
    for j in range(ncols):
        axs[-1, j].set_xlabel(r'$V_g$ in mV')
    # plt.tight_layout()
    if show:
        plt.show()
    return fig


def look_at_exp():
    """ This function allows to visualise each experimental data
        one diagram at a time. note that is actually creates a new exp
        dataloader from the global exp_dataset and does not use the global
        exp_dataloader"""
    dataloader = DataLoader(exp_dataset, batch_size=1, shuffle=False)
    infos = exp_dataset.info
    for diagram, label, index, mask in dataloader:
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
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        i = 0
        for k in range(3):
            for X, y, index, mask in dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                i += 1
    test_loss /= num_batches * 3
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss


def train_loop(dataloader, model, loss_fn, optimizer, gen=False):
    """ This function is essentially the same as in 'LazyEcMesure.py'.
        it is used in the 'test_GAN' function
        it loops over the dataloader and trains the model exactly 1 epoch.
        dataloader: the training data to use
        model: the model to train
        loss_fn: the loss function to minimise
        optimiser: the algorythm that will optimise the model
        gen: wether the model we are training is a generator or a discriminator"""
    size = len(dataloader.dataset)
    i = 0
    cron.start('train_print')
    running_loss = 0
    for batch, (X, y, index, mask) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # ---> Compute prediction and loss
        if gen:
            batch_ = int(X.size()[0])
            rand_vect = torch.normal(1, 0, size=[batch_, RANDOM_VECT_SIZE], device=device)
            pred = model(X, rand_vect)
        else:
            pred = model(X)
        loss = loss_fn(pred, y)
        running_loss += loss

        # ---> Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1

        # ---> printing timer
        if cron.t('train_print') >= 10.0:
            cron.start('train_print')
            current = (batch + 1) * len(X)
            temp = current / size
            bar = '[' + int(temp * 30) * '#' + int((1 - temp) * 30) * '-' + ']'
            print("loss: {loss:.5f}  {bar:>} {perc:.2%}".format(loss=running_loss / current, bar=bar, perc=temp))
    cron.stop('train_print')
    return running_loss / size


def GAN_train_loop(batch, model, loss_fn, optimizer, gen=False):
    """ This function is used in the 'train_GAN' function. it is used
    to train the GAN. unlike the 'train_loop' function, it does not
    loop over an entier epoch. instead, it trains the given model on
    exactly one batch, which is given as the first argument.
    NOTE: this function can train the generator and the discriminator.
    batch: the batch of data to train on. feed it as is, as a tupple with
        the arguments being: (X, y, index, mask)
    model: the model to train
    loss_fn: the loss function to minimise
    optimiser: the algorithm used to train the model
    gen: True if the model is a generator, False if it is a discriminator"""
    X, y, index, mask = batch
    size = len(index)

    X = X.to(device)
    y = y.to(device)
    # ---> Compute prediction and loss
    if gen:
        rand_vect = torch.normal(1, 0, size=[size, RANDOM_VECT_SIZE], device=device)
        pred = model(X, rand_vect)
    else:
        pred = model(X)
    loss = loss_fn(pred, y)

    # ---> Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def basicTrainingSequence(model, loss_fn, optimizer, train_dataloader, test_dataloader, numEpoch, init_epoch=1,
                          exp_dataloader=None):
    """ This function is the same as in the 'LazyEcMesure.py'. it is used in the
    test_GAN function, to verify that the generator generates data representative
    of the experimental data.
    this function takes care of evreything that needs to be done during training:
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
    # TODO continue run / checkpoint ???
    best = None
    best_summary = {}
    # ------------->>> training loop
    for E in range(numEpoch):
        print("Starting Epoch %s \n ========================================" % (E + 1))
        # ---> training a new epoch
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        loss = test_loop(test_dataloader, model, loss_fn)

        # ---> loging points to WandB
        log_dic = {"t_loss": loss, "t_train loss": train_loss, "t_epoch": E + init_epoch}
        exp_loss = None
        if exp_dataloader is not None:
            exp_loss = test_loop(exp_dataloader, model, loss_fn)
            log_dic["t_exp loss"] = exp_loss
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


def basicGANTrainingSequence(disc_model, disc_loss_fn, disc_optimizer,
                             gan_loss_fn, gan_optimizer, numEpoch, init_epoch=1):
    """ this function takes care of evrything that needs to be done during training
    of the GAN network:
    training, testing, logging to WandB, saving the model and etc.
    it will train the generator numEpoch epoch.

    disc_model: the discriminator model CLASS (not an instance). the class is necessarry
        as we reset the discriminator model several times during training
    disc_loss_fn: the loss function to minimise during training of the discriminator
    disc_optimiser: the disc optimiser CLASS (not an instance) during training
        of the discriminator. this is necessary to set the optimiser parameters during
        training when we reset the model
    gan_loss_fn: a custom loss function that takes in the prediction, the label and
        a discriminator model as input. it is used to evaluate the loss during the
        training of the generator
    gan_optimiser: the optimiser to use to optimise the generator
    numEpoch: the number of epoch to train for
    init_epoch: the model's starting epoch (used when loading a previously trained model)
    """

    global discriminators, gan_model
    discriminators = disc_model().to(device)
    for E in range(numEpoch):
        print("Starting Epoch %s \n ========================================" % (E + 1))
        # ---> reset discriminator every RESET_RATE epoch
        if E % RESET_RATE == 0:
            discriminators = disc_model().to(device)

        # ---> preparing to train 1 epoch
        i = 0
        cron.start('train_print')
        log_dic = {"epoch": E + init_epoch}

        gan_running_loss = 0
        disc_running_loss = 0
        size = len(disc_dataloader.dataset)
        disc_iter = iter(disc_dataloader)
        disc_batch = next(disc_iter, None)
        d_optimiser = disc_optimizer(discriminators.parameters())
        while disc_batch is not None:  # loop over the entirer disc_dataloader once
            # ---> training discriminator
            discriminators.zero_grad()
            disc_running_loss += GAN_train_loop(disc_batch, discriminators,
                                                disc_loss_fn, d_optimiser)

            # here we train the gen for "TRAIN_SEGMENTATION" batch after every "TRAIN_SEGMENTATION"
            # batch we trained the discriminator
            if (i+1) % TRAIN_SEGMENTATION == 0:
                loss_fn = lambda pred, y: gan_loss_fn(pred, y, discriminators)
                gan_model.zero_grad()
                for j in range(TRAIN_SEGMENTATION):
                    # ---> training generator
                    gan_running_loss += GAN_train_loop(next(iter(gan_dataloader)), gan_model, loss_fn, gan_optimizer, gen=True)

            disc_batch = next(disc_iter, None)
            i += 1
            # ---> progress printing (freezes when training discriminator)
            if cron.t('train_print') >= 10.0:
                cron.start('train_print')
                current = i * BATCH_SIZE
                temp = current / size
                bar = '[' + int(temp * 30) * '#' + int((1 - temp) * 30) * '-' + ']'
                print("loss: {loss:.5f}  {bar:>} {perc:.2%}".format(loss=disc_running_loss / current, bar=bar, perc=temp))
        # --------------> logging data to WandB
        cron.stop('train_print')
        log_dic['exp_loss'] = test_loop(exp_dataloader, discriminators, disc_loss_fn)
        log_dic['Disc_loss'] = disc_running_loss/size
        log_dic['gen_loss'] = gan_running_loss/size
        # ---> generating figures to see samples progression
        gen_fig = lookAtData(valid_dataloader, sim_datasets['valid'].info, 4, 8, show=False)
        log_dic["gen_fig"] = gen_fig
        wandb.log(log_dic)
        plt.close()  # free up the plots

        # ------------------------ SAVE_MODEL --------------------------
        model_scripted = torch.jit.script(gan_model)  # Export to TorchScript
        model_scripted.save(GEN_NETWORK_DIRECTORY + RUN_NAME + '_' + ID + '.pt')
    return


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
    # ---> finding the model's file
    for filename in os.listdir(NETWORK_DIRECTORY):
        f = os.path.join(NETWORK_DIRECTORY, filename)
        # checking if it is a file
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


def load_gan_model(model_name, configs=None, branch_training=True, tags=None, train=True):
    ''' This function allows to load previously trained generators
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
        train: wether or not load the model in train mode or eval mode. also,
            wether or not to initiate a new WandB run'''
    # TODO checkpoint resume
    global ID

    # ---------------->>> Loading the model
    model_file_id = None
    # ---> finding the model's file
    for filename in os.listdir(GEN_NETWORK_DIRECTORY):
        f = os.path.join(GEN_NETWORK_DIRECTORY, filename)
        if os.path.isfile(f):
            if filename.find(model_name) != -1:
                model_file_id = filename[0:filename.find('.')]
                break
    if model_file_id is None:
        raise 'model_name= %s not found in %s directory' % (model_name, GEN_NETWORK_DIRECTORY)
    # ---> actually loading the model
    model = torch.jit.load(GEN_NETWORK_DIRECTORY + model_file_id + '.pt')
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
        configs['gan_architecture'] = prev_config['gan_architecture']

    # ---> copying the previous ID if don't branch training
    if not branch_training:
        ID = model_file_id[model_file_id.find('_') + 1:]
        print('ID %s' % ID)
    else:
        configs['parent_run'] = model_name

    run = None
    # ---> Creating a new WandB run if in training mode
    if train:
        run = wandb.init(project=PROJECT, entity=ENTITY, id=ID, resume="allow",
                         config=configs, tags=tags)  # notes, tags, group  are other usfull parameters
        if branch_training:
            wandb.log({"exp_loss": summary['exp_loss'], "Disc_loss": summary['Disc_loss'], "epoch": init_epoch,
                       "gen_loss": summary['gen_loss']})
    return run, model, init_epoch + 1, configs


# ========================= CREATING A CUSTOM MODEL =========================
# ----------------------------- Discriminator -------------------------------
class Bottleneck(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, bias=True, stride=1):
        super(Bottleneck, self).__init__()
        # bias is usally False in resnet
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, bias=bias)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, bias=bias)
        self.batchNormLike = nn.InstanceNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_planes != out_planes or stride != 1:
            self.downsample = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)

        # self.dropout = nn.Dropout(0.15, inplace=True)
        pass

    def forward(self, x):
        out = self.conv1(x)
        # out = self.dropout(out)
        out = self.batchNormLike(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.dropout(out)
        out = self.batchNormLike(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.dropout(out)
        out = self.batchNormLike(out)

        # skip connection
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        # out = self.dropout(out)
        out = self.relu(out)
        return out

    pass


class ResLike1_0(nn.Module):
    def __init__(self):
        super(ResLike1_0, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.relu = nn.ReLU(inplace=True)
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
        x = self.lin1(x)
        return x


# ----------------------------- Generator -------------------------------
class FracConv(nn.Module):
    """ This is the basic building block of the generator network"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1):
        super(FracConv, self).__init__()
        # bias is usally False in resnet
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchNormLike = nn.InstanceNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        # self.dropout = nn.Dropout(0.15, inplace=True)
        pass

    def forward(self, x):
        out = self.conv1(x)
        # out = self.dropout(out)
        out = self.batchNormLike(out)
        out = self.relu(out)
        return out
    pass


class MyConv2D(nn.Module):
    """ This is another basic building block of the generator network"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1):
        super(MyConv2D, self).__init__()
        # bias is usally False in resnet
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchNormLike = nn.InstanceNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        # self.dropout = nn.Dropout(0.15, inplace=True)
        pass

    def forward(self, x):
        out = self.conv1(x)
        # out = self.dropout(out)
        out = self.batchNormLike(out)
        out = self.relu(out)
        return out

    pass


class GenNet1_0(nn.Module):
    """ the definition of the generator network. it takes a sample as input and
    a random vector of size RANDOM_VECT_SIZE"""
    def __init__(self):
        super(GenNet1_0, self).__init__()
        self.flatten = nn.Flatten()
        self.project = nn.Linear(RANDOM_VECT_SIZE, 4 * 4 * (1028 - BACK_FEED))  # then reshape this layer to 3D tensor
        self.fracConv1 = FracConv(1028, 512 - BACK_FEED, 4, stride=2, padding=1)
        self.fracConv2 = FracConv(512, 256 - BACK_FEED, 6, stride=2, padding=2)
        self.fracConv3 = FracConv(256, 128 - BACK_FEED, 6, stride=2, padding=2)
        self.fracConv4 = FracConv(128, 64 - BACK_FEED, 6, stride=2, padding=2)
        self.fracConvLast = nn.ConvTranspose2d(64, 1, 6, stride=2, padding=2)
        # self.fracConv1 = nn.ConvTranspose2d(1028, 512)

        self.conv1 = MyConv2D(1, BACK_FEED, 5, stride=2, padding=2)
        self.conv2 = MyConv2D(BACK_FEED, BACK_FEED, 5, stride=2, padding=2)
        self.conv3 = MyConv2D(BACK_FEED, BACK_FEED, 5, stride=2, padding=2)
        self.conv4 = MyConv2D(BACK_FEED, BACK_FEED, 5, stride=2, padding=2)
        self.conv5 = MyConv2D(BACK_FEED, BACK_FEED, 5, stride=2, padding=2)

        # self.batchNormLike = nn.InstanceNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rand_vect):
        # ---> deconstructing simulated data
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # ---> generating random vector and projecting it
        if len(x.size()) == 4:
            batch = int(x.size()[0])
            out = rand_vect
            out = self.project(out)
            out = out.view(batch, -1, 4, 4)
        else:
            out = rand_vect
            out = self.project(out)
            out = out.view(-1, 4, 4)
        out = torch.cat([x5, out], dim=-3)  # concatenate with deconstructed data
        out = torch.cat([x4, self.fracConv1(out)], dim=-3)  # apply fracConv and concatenate
        out = torch.cat([x3, self.fracConv2(out)], dim=-3)  # rinse and repeat...
        out = torch.cat([x2, self.fracConv3(out)], dim=-3)
        out = torch.cat([x1, self.fracConv4(out)], dim=-3)
        out = self.fracConvLast(out)
        return out


class GenNet2_0(nn.Module):
    """ a second declaration of the generator"""
    def __init__(self):
        super(GenNet2_0, self).__init__()
        self.flatten = nn.Flatten()
        self.project = nn.Linear(RANDOM_VECT_SIZE, 4 * 4 * (1028 - BACK_FEED))  # then reshape this layer to 3D tensor
        self.fracConv1 = FracConv(1028, 512, 4, stride=2, padding=1)
        self.fracConv2 = FracConv(512, 256, 6, stride=2, padding=2)
        self.fracConv3 = FracConv(256, 128, 6, stride=2, padding=2)
        self.fracConv4 = FracConv(128, 64, 6, stride=2, padding=2)
        self.fracConvLast = nn.ConvTranspose2d(64, 1, 6, stride=2, padding=2)
        # self.fracConv1 = nn.ConvTranspose2d(1028, 512)

        self.conv1 = MyConv2D(1, 128, 7, stride=2, padding=3)
        self.conv2 = MyConv2D(128, 512, 5, stride=2, padding=2)
        self.conv3 = MyConv2D(512, 1028, 5, stride=2, padding=2)
        self.conv4 = MyConv2D(1028, 512, 5, stride=2, padding=2)
        self.conv5 = MyConv2D(512, BACK_FEED, 5, stride=2, padding=2)

        # self.batchNormLike = nn.InstanceNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rand_vect):
        # deconstructing simulated data
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # generating random vector
        if len(x.size()) == 4:
            batch = int(x.size()[0])
            out = rand_vect
            out = self.project(out)
            out = out.view(batch, -1, 4, 4)
        else:
            out = rand_vect
            out = self.project(out)
            out = out.view(-1, 4, 4)
        out = torch.cat([x5, out], dim=-3)
        # generating new data
        # out = torch.cat([x4, self.fracConv1(out)], dim=-3)
        # out = torch.cat([x3, self.fracConv2(out)], dim=-3)
        # out = torch.cat([x2, self.fracConv3(out)], dim=-3)
        # out = torch.cat([x1, self.fracConv4(out)], dim=-3)
        out = self.fracConv1(out)
        out = self.fracConv2(out)
        out = self.fracConv3(out)
        out = self.fracConv4(out)
        out = self.fracConvLast(out)
        return out


# ---> custom loss function for the generator
def FOTloss(generated_sample, label, disc_model, loss_fn):  # my test loss function
    pred = disc_model(generated_sample)
    loss = loss_fn(pred, label)
    return loss


# ======================== MAIN ROUTINES ========================
def analise_network(model_name, datatype='valid'):
    """ same function as in LazyEcMesure. to be used on discriminator models"""
    dataloader = sim_dataloaders[datatype]
    infos = sim_datasets[datatype].info
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
    Vds_res = []
    Vg_res = []
    square_res = []
    for k in range(3):
        with torch.no_grad():
            for X, y, index, mask in dataloader:
                info = infos[index]
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                error.append(abs(float((pred - y) / y)) * 100.0)  # error in %
                loss.append(loss_fn(pred, y).item())
                alpha.append(info['ag'])
                Ec.append(info['Ec'])
                n_levels.append(len(info['levels']))
                T.append(info['T'])
                Vds_res.append(float(y))
                Vg_res.append(float(y / info['ag']))
                square_res.append(float(y ** 2 + (y / info['ag']) ** 2))
                if loss[-1] > 999999:
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

    wandb.log({"alpha fig": figAlpha,
               "Ec fig": figEc,
               "n level fig": figLv,
               "T fig": figT,
               "Vds_res_fig": figVds_res,
               "Vg_res_fig": figVg_res,
               "square_res_fig": figSquare_res, })
    return loss, alpha


def test_on_exp(model_name):
    """ same function as in LazyEcMesure. to be used on discriminator models"""
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
        for X, y, index, mask in dataloader:
            info = infos[index]
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
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

    wandb.log({"exp_perc": figExp_perc,
               "exp_abs": figExp_abs,
               "exp_guess": figExp_guess,
               "exp_raw_y": figExp_raw_y})
    return


def test_GAN():
    """ This function is verry similar to the 'train()' function in the
    'LazyEcMesure.py' file as it trains a neural network to find the Ec
    Values. I use this function to verify if my generated data allows
    me to converge on the experimental data"""
    # TODO implement parent run everywhere

    global ID, RUN_NAME, gan_model
    gan_model_name = "trim-shape"
    configs = {
        "disc_architecture": "ResLike2_0",
        "disc_loss_fn": "mean squared error loss",
        "disc_optimiser": "Adam",
        "gan_architecture": "GenNet1_0",
        "gan_model_name": gan_model_name,
        "epochs": 1000,
        "disc_lr": 1E-3,
        "disc_moment": 0.9,
        "batch_size": BATCH_SIZE,
        "data_used": "3.0 black_square_minimalist",
        "data_size": len(sim_dataloaders['train'].dataset),
        "valid_size": len(sim_dataloaders['valid'].dataset),
        "exp_data_size": len(exp_dataloader.dataset),
        "info": 'debug',
    }
    tags = ['ResLike2_0']
    print('Dataset train size = %s' % len(sim_datasets['train']))

    # ======================= BUILDING MODEL AND WANDB =======================
    # gan model
    trash1, gan_model, trash2, configs = load_gan_model(gan_model_name, configs, train=False)
    del trash1, trash2
    # fitting model
    model_name = None
    branch_training = True  # always True unless continuing a checkpoint
    if model_name is None:
        model = ResLike1_0()
        model = model.to(device)

        run = wandb.init(project=PROJECT, entity=ENTITY, id=ID, resume="allow",
                         config=configs, tags=tags, group="validation")  # notes, tags, group are other usfull parameters
        init_epoch = 1
    else:
        run, model, init_epoch, configs = load_model(model_name, configs, branch_training=branch_training, tags=tags)

    # ----------------->>> loss and optimisers
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), configs['disc_lr'], (configs['disc_moment'], 0.999))

    # ----------------->>> training the network
    RUN_NAME = run.name
    basicTrainingSequence(model, loss_fn, optimizer, gan_dataloader,
                          valid_dataloader, configs['epochs'], init_epoch=init_epoch,
                          exp_dataloader=exp_dataloader)


def train_GAN():
    """ Call this function in your main to train a network. it will run
    the training sequence based on the parameters you have set inside
    this function"""
    global ID, RUN_NAME, gan_model
    # =========================== CONFIGURATIONS =================================
    # NOTE: some configs are in the GLOBAL section at the top of the code
    configs = {
        "disc_architecture": "ResLike2_0",
        "disc_loss_fn": "mean squared error loss",
        "disc_optimiser": "Adam",
        "gan_architecture": "GenNet1_0",
        "gan_loss_fn": "FOTloss",
        "gan_optimiser": "Adam",
        "epochs": 1000,                # here, set the max #epoch you want to train for
        "gan_lr": 0.0002,
        "gan_moment": 0.5,
        "disc_lr": 0.0002,
        "disc_moment": 0.5,
        "batch_size": BATCH_SIZE,
        "reset_rate": RESET_RATE,
        "train_segm": TRAIN_SEGMENTATION,
        "data_used": "3.0 black_square_minimalist",  # small description
        "data_size": len(sim_dataloaders['train'].dataset),
        "valid_size": len(sim_dataloaders['valid'].dataset),
        "exp_data_size": len(exp_dataloader.dataset),
        "info": 'reset evrey 7, BF=16, 10 same',     # small description
    }
    tags = ['ResLike2_0', 'GenNet1_0']
    print('Dataset train size = %s' % len(sim_datasets['train']))

    # ======================= BUILDING MODEL AND WANDB =======================
    disc_model = ResLike1_0      # give the model class, not an instance
    # No need to enter full model name, just the 2 words does the tric. ex: 'kind-dawn' or 'eager-smoke'
    model_name = "trim-shape"    # put the name of the model you want to load, None to start a new model
    branch_training = True       # always True unless continuing a checkpoint
    if model_name is None:
        # ---> create a new model
        gan_model = GenNet1_0().to(device)

        run = wandb.init(project=PROJECT, entity=ENTITY, id=ID, resume="allow",
                         config=configs, tags=tags, group="debug")  # notes, tags, group are other usfull parameters
        init_epoch = 1
    else:
        # ---> load a model
        run, gan_model, init_epoch, configs = load_gan_model(model_name, configs, branch_training=branch_training, tags=tags)

    # ----------------->>> loss and optimisers
    disc_loss_fn = nn.MSELoss()
    disc_optimizer = lambda parameters:torch.optim.Adam(parameters, configs['disc_lr'], (configs['disc_moment'], 0.999))

    gan_loss_fn = lambda X, y, model_num: FOTloss(X, y, model_num, disc_loss_fn)
    gan_optimizer = torch.optim.Adam(gan_model.parameters(), configs['disc_lr'], (configs['disc_moment'], 0.999))

    # ----------------->>> training the network
    RUN_NAME = run.name
    basicGANTrainingSequence(disc_model, disc_loss_fn, disc_optimizer,
                             gan_loss_fn, gan_optimizer, configs['epochs'], init_epoch)
    return


# ============================ MAIN ============================
def main():
    # --------------->>> creating the dataloaders
    global exp_dataloader, sim_dataloaders, disc_dataloader, gan_dataloader, valid_dataloader
    exp_dataloader = DataLoader(exp_dataset, batch_size=BATCH_SIZE, shuffle=True)
    sim_dataloaders = {key: DataLoader(sim_datasets[key], batch_size=BATCH_SIZE, shuffle=True)
                       for key in sim_datasets}
    disc_dataloader = DataLoader(GenDataset(sim_datasets['train'], exp_dataset),
                                 batch_size=BATCH_SIZE, shuffle=True)
    gan_dataloader = DataLoader(GenDataset(sim_datasets['train']),
                                 batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(GenDataset(sim_datasets['valid']),
                                 batch_size=BATCH_SIZE, shuffle=True)

    # --------------->>> functions to uncomment depending on what you want to do
    # test_GAN()
    train_GAN()
    # get_input_stats(img_dataloaders['valid'], title='valid mean')
    # get_input_stats(exp_dataloader,  title='exp mean')

    # for i in range(10):
    # lookAtData(img_dataloaders['train'], img_datasets['train'].info, 4, 8)
    # look_at_exp()
    # train()
    # test_on_exp("eternal-feather")
    # analise_network("smart-wave", 'valid')


if __name__ == '__main__':
    main()
