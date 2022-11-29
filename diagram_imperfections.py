import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import beta
from math import ceil, floor
from numpy.random import randint, random
from scipy.ndimage import gaussian_filter
import opensimplex as simplex
import yaml

# ===================== DECLARING CONSTANTS ======================
DARK_THICKNESS = 20
BLACK_BOX = 200


def white_noise(sample, noise_scale):
    noise = np.random.normal(size=sample.shape)
    return sample + noise*noise_scale


def add_in_dark_current(sample):
    zeros = np.zeros([sample.shape[0], DARK_THICKNESS])
    return np.concatenate([zeros, sample], axis=1)


def pltBeta(a, b, loc=0.0, scale=1.0):
    ''' this function allows you to visualise a distribution function
    (not by using a statistical aproach)'''
    x = np.linspace(loc + 0.001*scale, loc + 0.999*scale, 100)
    plt.title(r"Beta distribution $\alpha$=%s, $\beta$=%s, loc=%s, scale=%s" %
              ('{:.2f}'.format(a), '{:.2f}'.format(b), '{:.2f}'.format(loc), '{:.2f}'.format(scale)))
    plt.plot(x, beta.pdf(x, a, b, loc=loc, scale=scale),
            'r-', lw=3, alpha=0.6, label='beta pdf')
    plt.ylim(bottom=0)
    plt.show()


def random_multiply(sample, min, max=None):
    if max is None:
        return sample * min
    width = max - min
    return sample * (min + np.random.uniform()*width)


def gaussian_blur(sample, target_info, min, max):
    Vg_res = abs(target_info["Vg_range"][1] - target_info["Vg_range"][0])/(target_info["Vg_range"][1]-1)  # mV/pix
    Vds_res = abs(target_info["Vds_range"][1] - target_info["Vds_range"][0])/(target_info["Vds_range"][1]-1)  # mV/pix
    sig_Vg = (random() * (max - min) + min)/Vg_res  # in x
    sig_Vds = (random() * (max-min) + min)/Vds_res  # in y
    return gaussian_filter(sample, [sig_Vds, sig_Vg])


def rand_current_modulation(sample, target_info, scale):
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


def gaussian_blob(x_pos, y_pos, means, stds):
    gauss = lambda x, mean, std: np.exp(-(x - mean) ** 2 / (2 * std * std))
    temp_x = gauss(x_pos, means[0], stds[0])
    temp_y = gauss(y_pos, means[1], stds[1])
    temp_x = np.repeat(temp_x[:, None], len(temp_y), axis=1)
    return temp_x * temp_y


def random_crop(sample, target_info):
    MIN_SIZE = 33
    box = target_info['box']
    # the floor() term in next line is for when we want to crop into the diamond
    temp = box[1][1] + floor((box[0][1] - box[1][1])/2 * 0.35)
    Vds_empty_space = int(beta.rvs(1.2, 0.80, 0, temp))
    # print(str(box[0][0] + 4) + '    ' + str(target_info['nVg']) + '    ' + str(Vds_empty_space) + '    ' + str(box[1][0]))
    newBox = [[randint(min([box[0][0]+4, target_info['nVg']-1]), target_info['nVg']), target_info['nVds'] - Vds_empty_space],
              [randint(0, box[1][0]), Vds_empty_space],
              ]

    # making surer the box satisfies a min size
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
    sample = sample[newBox[1][1]:newBox[0][1], newBox[1][0]:newBox[0][0]]
    # print("old: %s,\t new: %s" % (box, newBox))
    return sample, newBox


def clip_current(sample, low=None, high=None):
    if low is not None:
        sample[np.abs(sample) < low] = low * np.sign(sample[np.abs(sample) < low])
        sample[sample == 0] = low
    if high is not None:
        sample[np.abs(sample) > high] = high * np.sign(sample[np.abs(sample) > high])
    return sample


def threshold_current2(sample, target_info):
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
    # print(n_levels)
    # print(target_info['degens'])
    # next, vsat can start inside the first diamond or be at the \simeq end of the last diamond
    Vsat_range = [diamond_width*(3/2), diamond_width*(1/2 + (n_levels - 1))]
    thresh_I = calc_threshold_current(Vg, Vds, Vsat_range)
    return sample + thresh_I


def calc_threshold_current2(Vg, Vds, Vsat_range):
    """ This function is responsible to immitate the transistor transiting
    into cunduction mode as we hit the Vg threshold voltage."""
    # TODO it is possible I_rise is underestimated du to electron temperature in the experimental data
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
    I = I*temp  # here, we need to multyply by -Vd

    # re-adjusting the scale of the current
    I = I/np.exp(I_sat)  # trying a normalisation technique
    I = I*np.exp(beta.rvs(1.5, 1.5, 2.5, 1.5))
    # I_dark = np.exp(-32)
    # I[np.abs(I)<I_dark] = I_dark * np.sign(I[np.abs(I)<I_dark])
    # I = np.log10(np.abs(I))
    # print(I)
    # plt.title(V_sat)
    # plt.imshow(I, cmap='hot')
    # cbar = plt.colorbar(label='current in ??????')
    # plt.show()
    return I


def threshold_current(sample, target_info):
    Vmesh = create_Vmesh(sample, target_info)

    # Creating a slope:
    a = target_info['ag']
    b = target_info['s_ratio']
    up_slope = 1/(-a/(1-b*(1-a)))
    up_slope = beta.rvs(2.5, 1.5, up_slope*1.2, abs(up_slope*1.2))
    down_slope = 1/(a/(b*(1-a)))
    down_slope = beta.rvs(1.5, 2.5, 0, abs(down_slope*1.2))
    # print(up_slope)
    # print(down_slope)
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


def create_Vmesh(sample, target_info):
    '''this function creates a voltage space mesh in mV. Vmesh[1] are Vg and Vmesh[0] are Vds'''
    Vmesh = np.indices(sample.shape).astype('float32')
    temp = target_info['Vg_range']
    temp1 = target_info['nVg'] - 1
    Vmesh[1] = Vmesh[1] * (temp[1] - temp[0]) / temp1 + temp[0]
    temp = target_info['Vds_range']
    temp1 = target_info['nVds'] - 1
    Vmesh[0] = -(Vmesh[0] * (temp[1] - temp[0]) / temp1 + temp[0])
    return Vmesh


def shift_slopes(Vmesh, up_slope, down_slope):
    '''this function shifts Vmesh along the directions of the diamond slopes'''
    Vmesh[1][Vmesh[0]>0] = Vmesh[1][Vmesh[0]>0] - Vmesh[0][Vmesh[0]>0]*up_slope
    Vmesh[1][Vmesh[0]<0] = Vmesh[1][Vmesh[0]<0] - Vmesh[0][Vmesh[0]<0]*down_slope
    return Vmesh


def calc_threshold_current(Vmesh, Vsat_range):
    """ This function is responsible to immitate the transistor transiting
    into cunduction mode as we hit the Vg threshold voltage."""
    # TODO it is possible I_rise is underestimated du to electron temperature in the experimental data
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
    I = I*-Vmesh[0]  # here, we would need to multyply by -Vd, but simulator is eronous

    # re-adjusting the scale of the current
    I = I/np.exp(I_sat)
    I = I*np.exp(beta.rvs(1.5, 1.5, 2.5, 1.5))

    # I_dark = np.exp(-32)
    # I[np.abs(I)<I_dark] = I_dark * np.sign(I[np.abs(I)<I_dark])
    # I = np.log10(np.abs(I))
    # # print(I)
    # plt.title(V_sat)
    # plt.imshow(I, cmap='hot')
    # cbar = plt.colorbar(label='current in ??????')
    # plt.show()
    return I


def calc_pente(coords):
    return (coords[3] - coords[1])/(coords[2] - coords[0])


FILEPATH = "data/sim3_0/train/"
def load_index(index):
    f = open(FILEPATH + '_data_indexer.yaml', 'r')
    target_info = yaml.load(f, Loader=yaml.FullLoader)[index]
    sample = np.load(FILEPATH + target_info['f'] + ".npy")
    return sample, target_info


# ========================= 3D MAPPING ==========================
def sine_integral(freq, phase, direction, Vmesh):
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


def black_square(sample):
    new_sample = np.zeros([BLACK_BOX, BLACK_BOX])
    if sample.shape[0] > BLACK_BOX:
        # print("BIG BOY")
        temp = ceil(sample.shape[0]/2)
        sample = sample[int(temp-BLACK_BOX/2):int(temp+BLACK_BOX/2)]
    if sample.shape[1] > BLACK_BOX:
        # print("BIG BOY")
        sample = sample[:, :BLACK_BOX]
    shape = sample.shape
    temp = shape[0]/2
    new_sample[int(BLACK_BOX/2 - ceil(temp)):int(BLACK_BOX/2 + floor(temp)), 0:shape[1]] = sample
    return new_sample


# ============================ MAIN ==============================
def main():
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


    # Vmesh = create_Vmesh(np.zeros(shape), dummy_dic)
    # Vmesh = shift_slopes(Vmesh, -0.5, 0.3)
    # print(Vmesh)
    # calc_threshold_current(Vmesh, [39, 40])

    pass


if __name__ == '__main__':
    main()




I_rise_coords = [
    [435.6, -28.45, 468.9, -18.24],
    [452.0, -29.57, 462.1, -22.09],
    [454.3, -28.52, 489.2, -20.25],
    [461.4, -26.96, 485.0, -20.23],
    [429.5, -28.08, 465.7, -18.63],
    [381.2, -22.62, 409.7, -17.05],
]

I_slope_coords = [
    [497, -21.5, 560, -16.6],
    [507, -21.5, 560, -17.3],
    [515, -22.1, 560, -18.92],
    [520, -22, 560, -19.4],
    [496, -16.1, 575, -14.3],
    [484, -18.5, 550, -14.9],
]



