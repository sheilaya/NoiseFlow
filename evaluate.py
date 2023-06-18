import sys
sys.path.append('core1')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from utils import flow_viz
from utils import frame_utils

from noiseflow import NoiseFlow
from utils.utils import InputPadder, forward_interpolate
import cv2

@torch.no_grad()
def create_sintel_submission(model, args, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean']:
        test_dataset = datasets.MpiSintel(args, split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            x_list, (sequence, frame) = test_dataset[test_id]
            
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(x_list[0].shape)
            x_list_new=[]
            for n in range(args.num_frames):
                img=x_list[n]
                # print(img.shape)
                img=padder.pad(img[None].cuda())
                # print(img[0].shape)
                x_list_new.append(img[0])

            x_list_new=torch.stack(x_list_new, dim=1)
            # print(x_list_new.shape)
            img_pred, flow_low, flow_pr = model(x_list_new, iters=iters, flow_init=flow_prev, test_mode=True)
            img_pred = torch.squeeze(img_pred)
            #print("img_pred:",img_pred.shape,flow_pr[0].shape)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
            img_denoise = padder.unpad(img_pred).permute(1, 2, 0).cpu().numpy()
            # print("2flow:", flow.shape)

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))
            img_file = os.path.join(output_dir, 'frame%04d.png' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            cv2.imwrite(img_file, img_denoise)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, args, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean']:
        val_dataset = datasets.MpiSintel(args,split='training', dstype=dstype)
        epe_list = []
        epe_list1 = []
        epe_list2 = []
        # mask1_list = []
        # mask2_list = []
        epe12_all=0
        epe1_all=0
        epe2_all=0
        for val_id in range(len(val_dataset)):
            x_list, img_gt,flow_gt, _ = val_dataset[val_id]

            padder = InputPadder(x_list[0].shape)
            x_list_new=[]
            for n in range(args.num_frames):
                img=x_list[n]
                # print(img.shape)
                img=padder.pad(img[None].cuda())
                # print(img[0].shape)
                x_list_new.append(img[0])

            x_list_new=torch.stack(x_list_new, dim=1)


            img_pred, flow_low, flow_pr = model(x_list_new, iters=iters, test_mode=True)

            flow = padder.unpad(flow_pr[0]).cpu()

            mask1 = (flow_gt[0, :, :].abs() == 0) & (flow_gt[1, :, :].abs() == 0)
            mask2 = ~mask1

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()

            epe1 = epe * mask1
            epe2 = epe * mask2

            epe_list.append(epe.view(-1).numpy())

            epe_list1.append(epe1.view(-1).numpy())
            epe_list2.append(epe2.view(-1).numpy())

            mask1_list=mask1.view(-1).numpy()
            mask2_list=mask2.view(-1).numpy()

            number1 = np.sum(mask1_list != 0)
            number2 = np.sum(mask2_list != 0)

            epe12_all+=np.sum(epe.view(-1).numpy())/(number1+number2)

            epe1_all +=np.sum(epe1.view(-1).numpy())/number1
            epe2_all += np.sum(epe2.view(-1).numpy()) / number2
            print("=====", number1, number2, np.sum(epe1.view(-1).numpy()) / number1,np.sum(epe2.view(-1).numpy()) / number2, np.sum(epe.view(-1).numpy()) / (number1 + number2))

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)


        epe12 = epe12_all/  len(val_dataset)
        epe1 = epe1_all / len(val_dataset)
        epe2 = epe2_all / len(val_dataset)

        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, epe1: %f, epe2: %f, epe12: %f" % (dstype, epe, epe1, epe2, epe12))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--num_frames', type=int, nargs='+', default=11)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(NoiseFlow(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    create_sintel_submission(model.module,args, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module,args)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)


