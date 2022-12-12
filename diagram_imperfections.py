""" Author: Michael Bedard
this section of the code is where we declare diffrent types of noise
to be applied on our experimental and simulated data to do some
data augmentation
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import beta
from math import ceil, floor
from numpy.random import randint, random
from scipy.ndimage import gaussian_filter
import yaml

# ===================== DECLARING CONSTANTS ======================
DARK_THICKNESS = 20  # number of pixels to add on the left of image (not implemented)
BLACK_BOX = 128  # size of the image when we use the black_square function
FILEPATH = "data/sim3_0/train/"  # the filepath to use when loading files to debug


# ======================= DEVELOPING TOOLS =======================
def pltBeta(a, b, loc=0.0, scale=1.0):
    ''' this function allows you to visualise a the beta distribution curve.
    it is verry practical when setting a new distribution'''
    x = np.linspace(loc + 0.001*scale, loc + 0.999*scale, 100)
    plt.title(r"Beta distribution $\alpha$=%s, $\beta$=%s, loc=%s, scale=%s" %
              ('{:.2f}'.format(a), '{:.2f}'.format(b), '{:.2f}'.format(loc), '{:.2f}'.format(scale)))
    plt.plot(x, beta.pdf(x, a, b, loc=loc, scale=scale),
            'r-', lw=3, alpha=0.6, label='beta pdf')
    plt.ylim(bottom=0)
    plt.show()


def calc_pente(coords):
    """ using coords, this function calculates the slope between 2 points
    coords:(x1, y1, x2, y2) """
    return (coords[3] - coords[1])/(coords[2] - coords[0])


def load_index(index):
    """ this function allows us to load some experimental data in order to
    test our noise during development"""
    f = open(FILEPATH + '_data_indexer.yaml', 'r')
    target_info = yaml.load(f, Loader=yaml.FullLoader)[index]
    sample = np.load(FILEPATH + target_info['f'] + ".npy")
    return sample, target_info


# ======================== DATA TRANSFORMS =======================
def random_crop(sample, target_info):
    """ randomly crops an image around the first diamond. to do so, it relies on the 'box'
    information in the target_info. as opposed to diamond crop, it does not only crop around
    the vicinity of the first diamond, but can leave several other diamonds in the image.
    the values hard coded in this function are my best guess of what would work"""

    MIN_SIZE = 33  # this is a minimum crop size. it is necessarry when you are not
    #                using the 'black_square' function as some convolution networks crash
    #                with images that are to small
    box = target_info['box']

    # the floor() term in the next line is to ensure we can crop slightly into the diamond
    temp = box[1][1] + floor((box[0][1] - box[1][1])/2 * 0.35)
    Vds_empty_space = int(beta.rvs(1.2, 0.80, 0, temp))
    newBox = [[randint(min([box[0][0]+4, target_info['nVg']-1]), target_info['nVg']), target_info['nVds'] - Vds_empty_space],
              [randint(0, box[1][0]), Vds_empty_space],
              ]

    # making sure the box satisfies a min size
    temp = newBox[0][0] - newBox[1][0]
    if temp < MIN_SIZE:
        temp = ceil((MIN_SIZE - temp) / 2)
        newBox[0][0] += temp
        newBox[1][0] -= temp
        if newBox[1][0] < 0:
            newBox[0][0] -= newBox[1][0]
            newBox[1][0] = 0

    temp = newBox[0][1] - newBox[1][1]
    if temp < MIN_SIZE:
        temp = ceil((MIN_SIZE - temp) / 2)
        newBox[1][1] -= temp
        newBox[0][1] += temp
    # croping the image
    sample = sample[newBox[1][1]:newBox[0][1], newBox[1][0]:newBox[0][0]]
    return sample, newBox


def diamond_crop(sample, target_info):
    """ This function crops around the first diamond of a simulated data, leaving only this diamond
    in the final image

    as this function does not have a min size like 'random_crop', it must be used with the
    black_square function as some convolution networks crash with images that are to small"""
    box = target_info['box']

    # the floor() term in next line is for when we want to crop into the diamond
    temp = box[1][1] + floor((box[0][1] - box[1][1])/2 * 0.35)
    Vds_empty_space = int(beta.rvs(1.2, 0.80, 0, temp))

    max_Vds = randint(min([box[0][0], target_info['nVg']-1]),
                      min([target_info['nVg'], box[0][0]+(box[0][0]-box[1][0])/2]))
    min_Vds = randint(0, box[1][0])
    newBox = [[max_Vds, target_info['nVds'] - Vds_empty_space],
              [min_Vds, Vds_empty_space],
              ]

    # croping the image
    sample = sample[newBox[1][1]:newBox[0][1], newBox[1][0]:newBox[0][0]]
    return sample, newBox


def black_square(sample):
    """ this function takes an image with a random size and return's it as a
    zero padded image of size (BLACK_BOX x BLACK_BOX). the input image is aligned
    to the left of the output image and centered on the y axis. if input image is
    biger than output image, the input image is croped.

    this function returns the updated image and a mask that masks the original
    image. this mask can be discarded in most cases, but was used in the GAN"""
    new_sample = np.zeros([BLACK_BOX, BLACK_BOX])
    mask = np.full([BLACK_BOX, BLACK_BOX], False)

    # cropping input image if too big
    if sample.shape[0] > BLACK_BOX:
        temp = ceil(sample.shape[0]/2)
        sample = sample[int(temp-BLACK_BOX/2):int(temp+BLACK_BOX/2)]
    if sample.shape[1] > BLACK_BOX:
        sample = sample[:, :BLACK_BOX]

    shape = sample.shape
    temp = shape[0]/2
    mask[int(BLACK_BOX/2 - ceil(temp)):int(BLACK_BOX/2 + floor(temp)), 0:shape[1]] = True
    new_sample[int(BLACK_BOX/2 - ceil(temp)):int(BLACK_BOX/2 + floor(temp)), 0:shape[1]] = sample
    return new_sample, mask


def random_multiply(sample, min, max=None):
    """ this function randomly multiplies an image by a random number:
    max = None: if max is not given, than the sample is multiplied by the min value.
    max != None: else, a random value between min and max is chosen using a normal
    distribution"""
    if max is None:
        return sample * min
    width = max - min
    return sample * (min + np.random.uniform()*width)


def clip_current(sample, low=None, high=None):
    """ This function is super practical in order to assign a minimum and maximum ABSOLUTE value
    to the current.
    low: the minimum absolute value of the current
    high: the maximum absolute value of the current"""
    if low is not None:
        sample[np.abs(sample) < low] = low * np.sign(sample[np.abs(sample) < low])
        sample[sample == 0] = low
    if high is not None:
        sample[np.abs(sample) > high] = high * np.sign(sample[np.abs(sample) > high])
    return sample


def change_res(sample, target):
    """This function changes the resolution of an experimental data image"""
    # TODO: we should modify this function so it caps the resolution based on the box dimention, not on image dimention
    min_res = 20  # this value caps the min resolution amount
    shape = sample.shape
    Vg_res = randint(0, shape[0]//min_res + 1) + 1
    Vds_res = randint(0, shape[1]//min_res + 1) + 1
    new_sample = sample[randint(0, Vg_res)::Vg_res, randint(0, Vds_res)::Vds_res]
    new_target = target / Vds_res
    return new_sample, new_target


def add_in_dark_current(sample):
    """ not implemented yet. this function was suposed to add zeros on the left of a simulated
    image so we wouldn't have to simulate the part of the image with no current
    (since this would take time)"""
    zeros = np.zeros([sample.shape[0], DARK_THICKNESS])
    return np.concatenate([zeros, sample], axis=1)


# ============================ NOISES ============================
def white_noise(sample, noise_scale):
    " this function adds white noise to the image from a gaussian distribution"
    noise = np.random.normal(size=sample.shape)
    return sample + noise*noise_scale


def gaussian_blur(sample, target_info, min, max):
    """ This function applies a gaussian blur using a gaussian with an std taken from
    a normal distribution between min and max. min and max are given in mV"""
    Vg_res = abs(target_info["Vg_range"][1] - target_info["Vg_range"][0])/(target_info["Vg_range"][1]-1)  # mV/pix
    Vds_res = abs(target_info["Vds_range"][1] - target_info["Vds_range"][0])/(target_info["Vds_range"][1]-1)  # mV/pix
    sig_Vg = (random() * (max - min) + min)/Vg_res  # in x
    sig_Vds = (random() * (max-min) + min)/Vds_res  # in y
    return gaussian_filter(sample, [sig_Vds, sig_Vg])


def rand_current_modulation(sample, target_info, scale):
    """ this function aplies random current modulations by multiplying the sample
    with random gaussian blobs. the parameters were empiricly chosen.
    scale: maximum amplitude of the deformations"""
    std_dist = lambda: beta.rvs(1, 1, 5, 70)
    amplitude_dist = lambda: beta.rvs(1.4, 1.4, -1, 2)
    x_width = target_info['Vg_range'][1] - target_info['Vg_range'][0]
    y_width = target_info['Vds_range'][1] - target_info['Vds_range'][0]
    area = x_width * y_width
    x_pos = np.linspace(0, x_width, target_info['nVg'])
    y_pos = np.linspace(0, y_width, target_info['nVds'])

    x_mean_dist = lambda: beta.rvs(1, 1, -x_width/2, 2*x_width)
    y_mean_dist = lambda: beta.rvs(1, 1, -y_width/2, 2*y_width)
    num_gausse = randint(floor(area/(50*50)), ceil(area/(20*20)))
    modulation = np.zeros(sample.shape)
    for i in range(num_gausse):
        means = [x_mean_dist(), y_mean_dist()]
        stds = [std_dist(), std_dist()]
        modulation += amplitude_dist() * gaussian_blob(x_pos, y_pos, means, stds).T
    # scaling the noise
    min = modulation.min()
    max = modulation.max()
    if abs(min) > 1 or max > 1:
        modulation /= np.max([abs(min), max])
    modulation = scale*modulation + 1
    return sample * modulation


def rand_current_addition(sample, target_info, scale):
    """ identical to 'rand_current_modulation' excepts it ads the gaussian blobs
    instead of multiplying them
    scale: maximum amplitude of the deformations"""
    std_dist = lambda: beta.rvs(1, 1, 5, 130)
    amplitude_dist = lambda: beta.rvs(1.4, 1.4, -1, 2)
    x_width = target_info['Vg_range'][1] - target_info['Vg_range'][0]
    y_width = target_info['Vds_range'][1] - target_info['Vds_range'][0]
    area = x_width * y_width
    x_pos = np.linspace(0, x_width, target_info['nVg'])
    y_pos = np.linspace(0, y_width, target_info['nVds'])

    x_mean_dist = lambda: beta.rvs(1, 1, -x_width/2, 2*x_width)
    y_mean_dist = lambda: beta.rvs(1, 1, -y_width/2, 2*y_width)
    num_gausse = randint(floor(area/(150*150)), ceil(area/(40*40)))
    modulation = np.zeros(sample.shape)
    for i in range(num_gausse):
        means = [x_mean_dist(), y_mean_dist()]
        stds = [std_dist(), std_dist()]
        modulation += amplitude_dist() * gaussian_blob(x_pos, y_pos, means, stds).T
    # scaling the noise
    min = modulation.min()
    max = modulation.max()
    if abs(min) > 1 or max > 1:
        modulation /= np.max([abs(min), max])
    return sample + scale*modulation


def threshold_current(sample, target_info):
    """ This function is almost identical to 'threshold_current2()' except
    it also shifts the Vds/Vg slope of the threshold voltage by a random amount to look
    more like experimental data"""
    Vmesh = create_Vmesh(sample, target_info)

    # Creating a slope:
    a = target_info['ag']
    b = target_info['s_ratio']
    up_slope = 1/(-a/(1-b*(1-a)))
    up_slope = beta.rvs(2.5, 1.5, up_slope*1.2, abs(up_slope*1.2))
    down_slope = 1/(a/(b*(1-a)))
    down_slope = beta.rvs(1.5, 2.5, 0, abs(down_slope*1.2))
    # shifting the Vmesh
    Vmesh = shift_slopes(Vmesh, up_slope, down_slope)

    # defining the possible range of Vsat
    diamond_width = target_info['Ec']/target_info['ag']
    n_levels = np.array(target_info['degens']).sum()
    MAX_LEVELS = 4
    if n_levels > MAX_LEVELS:
        n_levels = MAX_LEVELS

    # next, vsat can start inside the first diamond or be at the \simeq end of the last diamond
    Vsat_range = [diamond_width*(3/2), diamond_width*(1/2 + (n_levels - 1))]
    thresh_I = calc_threshold_current(Vmesh, Vsat_range)
    return sample + thresh_I


def threshold_current2(sample, target_info):
    """ this function applies a threshold current to the sample, meaning it tries
    to loosly replicate what hapens when the transistor starts to be conductive
    It was created  empirically by looking at Antoin's limited data"""
    temp = target_info['Vg_range']
    Vg = np.linspace(temp[0], temp[1], target_info['nVg'])
    temp = target_info['Vds_range']
    Vds = np.linspace(temp[0], temp[1], target_info['nVds'])

    # defining the possible range of Vsat
    diamond_width = target_info['Ec']/target_info['ag']
    n_levels = np.array(target_info['degens']).sum()
    MAX_LEVELS = 5
    if n_levels > MAX_LEVELS:
        n_levels = MAX_LEVELS

    # next, vsat can start inside the first diamond or be at the \simeq end of the last diamond
    Vsat_range = [diamond_width*(3/2), diamond_width*(1/2 + (n_levels - 1))]
    thresh_I = calc_threshold_current(Vg, Vds, Vsat_range)
    return sample + thresh_I


# ===================== BACKGROUND FUNCTIONS =====================
def gaussian_blob(x_pos, y_pos, means, stds):
    """ This function calculates the height of a gaussian blob given its parameters.
    x_pos: a 1D array of all x positions to calculate
    y_pos: a 1D array of all y positions to calculate
    means: tuple of 2 elements: (x_mean, y_mean)
    stds: tuple of 2 elements: (x_std, y_std)
    return: a 2D array of the 2D gaussian blob's amplitude"""
    gauss = lambda x, mean, std: np.exp(-(x - mean) ** 2 / (2 * std * std))
    temp_x = gauss(x_pos, means[0], stds[0])
    temp_y = gauss(y_pos, means[1], stds[1])
    temp_x = np.repeat(temp_x[:, None], len(temp_y), axis=1)
    return temp_x * temp_y


def create_Vmesh(sample, target_info):
    '''this function creates a voltage space mesh in mV. Vmesh[1] are Vg and Vmesh[0] are Vds.
    it is similar to np.indicies() but instead of indicies, each values represent the applied
    Vg and Vds voltage of each pixel'''
    Vmesh = np.indices(sample.shape).astype('float32')
    temp = target_info['Vg_range']
    temp1 = target_info['nVg'] - 1
    Vmesh[1] = Vmesh[1] * (temp[1] - temp[0]) / temp1 + temp[0]
    temp = target_info['Vds_range']
    temp1 = target_info['nVds'] - 1
    Vmesh[0] = -(Vmesh[0] * (temp[1] - temp[0]) / temp1 + temp[0])
    return Vmesh


def shift_slopes(Vmesh, up_slope, down_slope):
    '''this function shifts Vmesh along the directions of the diamond slopes
    This it what alows us to shift the slopes of the threshold current
    up_slope: the slope of to use when Vds > 0
    down_slope: the slope to use when Vds < 0'''
    Vmesh[1][Vmesh[0]>0] = Vmesh[1][Vmesh[0]>0] - Vmesh[0][Vmesh[0]>0]*up_slope
    Vmesh[1][Vmesh[0]<0] = Vmesh[1][Vmesh[0]<0] - Vmesh[0][Vmesh[0]<0]*down_slope
    return Vmesh


def calc_threshold_current(Vmesh, Vsat_range):
    """This function is almost identical to 'calc_threshold_current2()' except
    it also shifts the Vds/Vg slope of the threshold voltage by a random amount to look
    more like experimental data
    instead of using 1D arrays for Vds and Vg, it uses a Vmesh as input. this is
    how we are able to angle the saturation curves: by manipulating the Vmesh
    with 'shift_slopes' """
    # setting the parameters with empirical random values
    Vsat_range = np.array(Vsat_range)
    I_rise = beta.rvs(1.6, 5.5, 0.15, 0.85)
    I_slope = beta.rvs(2.2, 3.0, 0.015, 0.085)
    I_sat = beta.rvs(3.7, 2.2, -25, 8.5)
    V_sat = beta.rvs(2.7, 1.5, loc=Vsat_range.min(), scale=np.abs(Vsat_range[1] - Vsat_range[0])) + (I_sat+30)/I_rise

    I_mask = Vmesh[1] < V_sat

    I = np.zeros(Vmesh[1].shape)
    I[I_mask] = (Vmesh[1][I_mask] - V_sat) * I_rise + I_sat
    I[I_mask == False] = (Vmesh[1][I_mask == False] - V_sat) * I_slope + I_sat

    I = np.exp(I)
    I = I*-Vmesh[0]

    # re-adjusting the scale of the current
    I = I/np.exp(I_sat)
    I = I*np.exp(beta.rvs(1.5, 1.5, 2.5, 1.5))
    return I


def calc_threshold_current2(Vg, Vds, Vsat_range):
    """ This function is responsible to immitate the transistor transiting
    into cunduction mode as we hit the Vg threshold voltage.
    It was empirically created by studying current curves as the transistor
    entered conduction mode
    """
    # setting the parameters with empirical random values
    Vsat_range = np.array(Vsat_range)
    I_rise = beta.rvs(1.6, 5.5, 0.15, 0.85)
    I_slope = beta.rvs(2.2, 3.0, 0.015, 0.085)
    I_sat = beta.rvs(3.7, 2.2, -25, 8.5)
    V_sat = beta.rvs(2.7, 1.5, loc=Vsat_range.min(), scale=np.abs(Vsat_range[1] - Vsat_range[0])) + (I_sat+30)/I_rise
    I_mask = Vg < V_sat

    I = np.zeros(len(Vg))
    I[I_mask] = (Vg[I_mask] - V_sat) * I_rise + I_sat
    I[I_mask == False] = (Vg[I_mask == False] - V_sat) * I_slope + I_sat

    I = np.tile(I, [len(Vds), 1])
    I = np.exp(I)
    temp = np.tile(Vds[:, None], [1, len(Vg)])
    I = I*temp

    # re-adjusting the scale of the current
    I = I/np.exp(I_sat)
    I = I*np.exp(beta.rvs(1.5, 1.5, 2.5, 1.5))
    return I


# ========================= 3D MAPPING ==========================
# was an experiment, but coudn't get it to work as intended (more time needed)
def sine_integral(freq, phase, direction, Vmesh):
    """ this function calculates the amplitude of a sinewave
    in 3D space at every point in Vmesh. it outputs a 2D plane
    of values."""
    X = np.dot(Vmesh, direction)
    # print(X.shape)
    temp = np.real(np.exp(1j *(phase + freq * 2*np.pi * X)))
    temp += Vmesh[:, :, 0] * direction[0]
    # temp += np.abs(X)
    # print(direction[0])
    # temp += X
    # plt.imshow(X, cmap='binary')
    # cbar = plt.colorbar(label='')
    # plt.show()
    return temp


def low_freq_3DMAP(sample, target_info):
    """ This is a function that maps each pixel of a sample
    to a diffrent value based on its current value and the
    Vds ann Vg of that sample. I sadly didn't passed enough
    time on this function as it somtimes erase almost every
    detail of the image so I coudn't get it to fully work as
    intended. the parameters are empirically found, and the main
    parameter to adjust here is the 'I_scale' value"""
    Vg = target_info['Vg_range']
    Vd = target_info["Vds_range"]
    # print(target_info['T'])
    Vmesh = create_Vmesh(sample, target_info)
    # I_scale = (random()*0.1)**2  # scale with harmonics
    I_scale_func = lambda T: 8E-5 * T * T + 0.002216 * T + 0.002
    I_scale = I_scale_func(target_info['T'])
    Vmesh = np.concatenate([sample[None, :, :]*I_scale, Vmesh], axis=0)
    Vmesh = np.moveaxis(Vmesh, 0, -1)
    # print(Vmesh.shape)
    harmonics = 10
    new_sample = np.zeros(sample.shape)
    for i in range(harmonics):
        minF, maxF = 1/1600, 1/40
        # converting those to octaves
        minF, maxF = np.log2(minF), np.log2(maxF)
        freq = 2**(random()*(maxF - minF) + minF)
        # print(1/freq)
        phase = random()*2*np.pi
        temp = random()*2*np.pi
        direction = np.array([random(), np.cos(temp), np.sin(temp)])
        direction /= np.linalg.norm(direction)
        amplitude = random()/freq/800
        new_sample += amplitude*sine_integral(freq, phase, direction, Vmesh)

    # new_sample = sample
    # normalise
    min_old, max_old = sample.min(), sample.max()
    # print(min_old - max_old)
    min_new, max_new = new_sample.min(), new_sample.max()
    new_sample = (new_sample - min_new)/(max_new-min_new)
    new_sample = new_sample*(max_old-min_old) + min_old
    # plt.imshow(new_sample, extent=[Vg[0],Vg[-1],Vd[0],Vd[-1]], cmap='binary_r')
    # cbar = plt.colorbar(label='')
    # plt.show()
    return new_sample


# ============================ MAIN ==============================
def main():
    """ This is just bits of codes I use to test certain functions. it has
    no other uses"""
    # pltBeta(2.5, 1.5)
    print('starting')
    shape = [100, 100]
    target_info = {'Vg_range': [0, 100],
                 'nVg': shape[1],
                 'Vds_range': [-50, 50],
                 'nVds': shape[0]}
    sample = np.zeros(shape)
    for j in range(15):
        sample, target_info = load_index(randint(0, 5500))  # load_index(13)
        sample = np.log(np.abs(clip_current(sample, 2E-14)))
        # plt.imshow(sample)
        # plt.show()
        for i in range(3):
            print('here')
            low_freq_3DMAP(sample, target_info)
    pass


if __name__ == '__main__':
    main()



