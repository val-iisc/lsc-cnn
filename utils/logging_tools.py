"""
utils.py: Some useful functions for LSC-CNN.
Authors : svp & muks & dbs
"""

import torch
import numpy as np
import datetime
import cv2
import matplotlib.pyplot as plt
import os



def log(f, txt, do_print=1):
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f.write(txt + '\n')

# Get the filename for the model stored after 'epochs_over' epochs got over
def get_filename(net_name, epochs_over):
    return net_name + "_epoch_" + str(epochs_over) + ".pth"


def load_net(networks, netfns, idx, fdir, name, set_epoch=True):
    net = networks

    filepath = os.path.join(fdir, name)
    print("Loading file...", filepath)

    if not os.path.isfile(filepath):
        print("Checkpoint file" + filepath + " not found!")
        raise IOError

    # print('=> loading checkpoint "{}"'.format(filepath))
    checkpoint_1 = torch.load(filepath)

    if set_epoch:
        try:
            args.start_epoch = checkpoint_1['epoch']
        except NameError:
            pass
    net.load_state_dict(checkpoint_1['state_dict'])

    # if netfns is not None:
    #   netfns.optimizers[idx].load_state_dict(checkpoint_1['optimizer'])

    print("=> loaded checkpoint '{}' ({} epochs over)".format(filepath, checkpoint_1['epoch']))

    if netfns is not None:
        return net, netfns
    else:
        return net

def save_checkpoint(state, fdir, name='checkpoint.pth'):
    filepath = os.path.join(fdir, name)
    torch.save(state, filepath)

def print_graph(maps, title, save_path):
    fig = plt.figure()
    st = fig.suptitle(title)
    for i, (map, args) in enumerate(maps):
        plt.subplot(1, len(maps), i + 1)
        if len(map.shape) > 2 and map.shape[0] == 3:
            plt.imshow(map.transpose((1, 2, 0)).astype(np.uint8),aspect='equal', **args)
        else:
            plt.imshow(map, aspect='equal', **args)
            plt.axis('off')
    plt.savefig(save_path + ".png", bbox_inches='tight', pad_inches = 0)
    fig.clf()
    plt.clf()
    plt.close()


