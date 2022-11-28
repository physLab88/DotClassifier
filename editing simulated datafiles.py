''' This code is to do some quick adjustments on the simulated data.
ex: remove certain files from the dataset'''

import numpy as np
import yaml
import os
import matplotlib.pyplot as plt

root_dir = "data/sim3_0/train/"


def verify_if_files_attached(delete_unattached=False):
    """ This function checks if there is a file attached to the main yalm file, which can occasionaly occure
    if you stoped the generating program prematurely"""
    f = open(root_dir + '_data_indexer.yaml', 'r')
    infos = yaml.load(f, Loader=yaml.FullLoader)
    # infos.pop(143)
    fail=0
    failed_index = []
    for info, index in zip(infos, range(len(infos))):
        try:
            print(info['f'])
        except:
            print("========= FAIL ===========")
            print(info)
            fail += 1
            failed_index.append(index)
    print(fail)
    print(failed_index)
    print(len(infos))
    if fail == 0 and delete_unattached:
        f = open(root_dir + '_data_indexer.yaml', 'w')
        yaml.dump(infos, f)


def verify_min_img_size(min_size=35, del_too_small=False):
    """ This function was created to remove files from the simulated dataset that have too small widths or height
    min_size:  minimum height and width size
    del_too_small: False if you just want to count those files, True if you want to permanently remove them
                   from the dataset
    """
    f = open(root_dir + '_data_indexer.yaml', 'r')
    infos = yaml.load(f, Loader=yaml.FullLoader)
    print(len(infos))
    fail=0
    failed_index = []
    for info, index in zip(infos, range(len(infos))):
        print(index)
        if info["nVds"] <= min_size or info['nVg'] <= min_size:
            fail += 1
            failed_index.append(index)
            print('Ec' + str(info['Ec']))
    print(fail)
    print(failed_index)
    # deleting files
    if del_too_small:
        files_to_delete = []
        for index in failed_index[::-1]:
            info = infos.pop(index)
            sample_name = root_dir + info['f'] + '.npy'
            files_to_delete.append(sample_name)

        fail = 0
        failed_index = []
        for info, index in zip(infos, range(len(infos))):
            if info["nVds"] <= min_size or info['nVg'] <= min_size:
                fail += 1
                failed_index.append(index)
                print('Ec' + str(info['Ec']))
        print(fail)
        print(failed_index)
        if fail == 0:
            for file in files_to_delete:
                os.remove(file)
            f = open(root_dir + '_data_indexer.yaml', 'w')
            yaml.dump(infos, f)
    return


def crop_image():
    """ This function is used to manually crop images. mostly used to crop experimental data"""
    f = open(root_dir + '_data_indexer.yaml', 'r')
    infos = yaml.load(f, Loader=yaml.FullLoader)
    print(len(infos))
    for index in range(len(infos)):
        info = infos[index]
        sample_name = root_dir + info['f'] + '.npy'
        diagram = np.load(sample_name)

        UL = [0, 0]
        LR = [-1, -1]
        while True:
            plt.imshow(np.log(np.abs(diagram[UL[1]:LR[1], UL[0]:LR[0]])), cmap='hot')
            plt.show()
            is_good = input("Is the diagram good? (y/n/c):  ")
            if is_good == 'y':
                Vg = np.linspace(info['Vg_range'][0], info['Vg_range'][1], info['nVg'])
                Vds = np.linspace(info['Vds_range'][0], info['Vds_range'][1], info['nVds'])
                Vg = np.concatenate([Vg, [0]])[UL[0]: LR[0]]
                Vds = np.concatenate([Vds, [0]])[UL[1]: LR[1]]
                if len(Vg) < 33:
                    print('error, Vg does not have enough pixels')
                elif len(Vds) < 33:
                    print('error, Vds does not have enough pixels')
                else:
                    info['Vg_range'] = [float(Vg[0]), float(Vg[-1])]
                    info['nVg'] = len(Vg)
                    info['Vds_range'] = [float(Vds[0]), float(Vds[-1])]
                    info['nVds'] = len(Vds)
                    diagram = diagram[UL[1]:LR[1], UL[0]:LR[0]]
                    # save everything
                    np.save(sample_name, diagram)
                    f = open(root_dir + '_data_indexer.yaml', 'w')
                    yaml.dump(infos, f)
                    print('yay')
                    break
            elif is_good == 'c':
                # c is for cancel, won't save or change anything of this datapoint
                break
            uper_left = input("uper left corner coords:  ")
            uper_left = uper_left.split()
            UL = [0, 0]
            for i, coord in zip(range(2), uper_left):
                if coord != '-':
                    print('changing coord')
                    UL[i] = int(coord)

            lower_right = input("lower right corner coords: ")
            lower_right = lower_right.split()
            LR = [-1, -1]
            for i, coord in zip(range(2), lower_right):
                if coord != '-':
                    print('changing coord')
                    LR[i] = int(coord)
            print(UL)
            print(LR)
    return


def main():
    # verify_if_files_attached(delete_unattached=False)
    verify_min_img_size(del_too_small=False)
    # crop_image()


if __name__ == '__main__':
    main()
