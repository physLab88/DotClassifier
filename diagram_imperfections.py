import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import beta
from math import ceil
from numpy.random import randint


# ===================== DECLARING CONSTANTS ======================
"""
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
"""


def random_crop(sample, target_info):
    MIN_SIZE = 33
    box = target_info['box']
    temp = randint(0, box[1][1] - 4)
    newBox = [[randint(box[0][0] + 4, target_info['nVg']), target_info['nVds'] - temp],
              [randint(0, box[1][0]), temp],  # uper left corner
              ]  # lower right

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


def threshold_current(sample, target_info):
    pass

def calc_threshold_current(Vg, Vds, Vsat_range):
    """ This function is responsible to immitate the transistor transiting
    into cunduction mode as we hit the Vg threshold voltage."""
    Vsat_range = np.array(Vsat_range)
    I_rise = beta.rvs(1.6, 5.5, 0.15, 0.85)
    I_slope = beta.rvs(2.2, 3.0, 0.015, 0.085)
    I_sat = beta.rvs(3.7, 2.2, -25, 8.5)
    V_sat = beta.rvs(2.7, 1.5, loc=Vsat_range.min(), scale=np.abs(Vsat_range[1] - Vsat_range[0]))
    I_mask = Vg < V_sat
    #I_dark = np.exp(-32)

    I = np.zeros(len(Vg))
    I[I_mask] = (Vg[I_mask] - V_sat) * I_rise + I_sat
    I[I_mask == False] = (Vg[I_mask == False] - V_sat) * I_slope + I_sat

    I = np.tile(I, [len(Vds), 1])
    I = np.exp(I)
    temp = np.tile(Vds[:, None], [1, len(Vg)])
    I = I*temp

    #I[np.abs(I)<I_dark] = I_dark * np.sign(I[np.abs(I)<I_dark])
    #I = np.log10(np.abs(I))
    #print(I)
    #plt.title(V_sat)
    #plt.imshow(I, cmap='hot')
    #cbar = plt.colorbar(label='current in ??????')
    #plt.show()
    return I


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


def calc_pente(coords):
    return (coords[3] - coords[1])/(coords[2] - coords[0])


# ============================ MAIN ==============================
def main():
    # pltBeta(2.7, 1.5)
    for coords in I_slope_coords:
        print(calc_pente(coords))
    Vg = np.linspace(0, 100, 100)
    Vds = np.linspace(-15, 15, 100)
    for i in range(30):
        threshold_current(Vg, Vds, [40, 70])
    pass


if __name__ == '__main__':
    main()






