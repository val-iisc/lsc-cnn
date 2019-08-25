"""
game.py: Torch implementation of GAME Metric
Authors : svp
"""
import numpy as np
import torch
'''
    Calculates GAME Metric as mentioned by Guerrero-Gómez-Olmedo et al. in 
    Extremely Overlapping Vehicle Counting, in IbPRIA, 2015.

    Parameters:
    -----------
    level - (int) level of GAME metric.
    gt - (np.ndarray) binary map of ground truth (HXW)
    pred - (np.ndarray) binary map of predictions (HXW)

    Returns
    -------
    mae - GAME for the level mentioned in the input.
'''
def game_metric(level, gt, pred):
    assert(gt.shape == pred.shape)
    num_cuts = np.power(2, level)
    num_blocks = np.power(4, level)
    h, w = gt.shape
    mae_normal = (np.abs(np.sum(gt) - np.sum(pred)))
    gt_reshape_torch = torch.from_numpy(gt)
    pred_reshape_torch = torch.from_numpy(pred)
    stride = int(h/float(num_cuts))
    gt_reshape_torch = gt_reshape_torch.unfold(0,  int(h/float(num_cuts)), stride).unfold(1, int(h/float(num_cuts)), stride)
    pred_reshape_torch = pred_reshape_torch.unfold(0,  int(h/float(num_cuts)), stride).unfold(1, int(h/float(num_cuts)), stride)
    
    gt_sum = torch.sum(torch.sum(gt_reshape_torch, dim=3), dim=2)
    pred_sum = torch.sum(torch.sum(pred_reshape_torch, dim=2), dim=2)
    gt_sum = gt_sum.data.numpy()
    pred_sum = pred_sum.data.numpy()
    mae = np.sum(np.abs(gt_sum - pred_sum))
    return mae
'''
    Wrapper for calculating GAME Metric

    Parameters:
    -----------
    gt - (np.ndarray) binary map of ground truth (HXW)
    pred - (np.ndarray) binary map of predictions (HXW)

    Returns
    -------
    mae_l1, mae_l2, mae_l3 - GAME for 3 different levels
'''
def find_game_metric(gt, pred):
    mae_normal_1 = np.abs(np.sum(gt) - np.sum(pred))
    h, w = gt.shape
    # -- Padding to make the size similar
    add_h = 0
    add_w = 0
    if h %8 != 0:
        add_h = h % 8
    if w % 8 !=0:
        add_w = w %8
    
    gt = np.resize(gt, (h-add_h, w-add_w))
    pred = np.resize(pred, (h-add_h, w-add_w))
    mae_another = np.abs(np.sum(gt) - np.sum(pred))

    mae_l1 = game_metric(1, gt, pred)
    mae_l2 = game_metric(2, gt, pred)
    mae_l3 = game_metric(3, gt, pred)

    return mae_l1, mae_l2, mae_l3
