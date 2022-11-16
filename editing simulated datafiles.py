''' This code is to do some quick adjustments on the simulated data.
ex: remove certain files from the dataset'''

import numpy as np
import yaml
import os


root_dir = "data/sim3_0/train/"


def verify_if_files_attached(delete_unattached=False):
    """ This function checks if there is a file attached to the main yalm file, which can occasionaly occure
    if you stoped the generating program prematurely"""
    f = open(root_dir + '_data_indexer.yaml', 'r')
    infos = yaml.load(f, Loader=yaml.FullLoader)
    infos.pop(143)
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


def main():
    # verify_if_files_attached(delete_unattached=False)
    verify_min_img_size(del_too_small=False)


if __name__ == '__main__':
    main()
