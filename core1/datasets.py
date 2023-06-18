# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, args, aug_params=None, sparse=False):
        self.num_frames = args.num_frames
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.gt_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            x_list=[]
            for n in range(self.num_frames):
                img=frame_utils.read_gen(self.image_list[index][n])

                if len(np.array(img).shape)==2:
                    img = np.stack((np.array(img).astype(np.uint8),) * 3, axis=-1)
                else:
                    img = np.array(img)
                img = torch.from_numpy(img).permute(2, 0, 1).float()
                x_list.append(img)

            return x_list, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])
        flow = np.array(flow).astype(np.float32)

        x_list=[]
        for n in range(self.num_frames):
            img = frame_utils.read_gen(self.image_list[index][n])
            img = np.array(img).astype(np.uint8)
            if len(img.shape) == 2:
                img = np.tile(img[..., None], (1, 1, 3))
            else:
                img = img[..., :3]
            x_list.append(img)
        #print("---------len:",len(self.image_list),len(self.flow_list),len(self.gt_list))
        img_gt = frame_utils.read_gen(self.gt_list[index])
        img_gt = np.array(img_gt).astype(np.uint8)
        if len(img_gt.shape) == 2:
            img_gt = np.tile(img_gt[..., None], (1, 1, 3))
        else:
            img_gt = img_gt[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                x_list, img_gt,flow = self.augmentor(x_list,img_gt, flow)

        for n in range(self.num_frames):
            img=x_list[n]
            x_list[n]=torch.from_numpy(img).permute(2, 0, 1).float()

        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        img_gt = torch.from_numpy(img_gt).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return torch.stack(x_list, dim=0), img_gt, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.gt_list = v * self.gt_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, args, aug_params=None, split='training', root='D:\\DST-Net\\generatedata\\10580\\RAFT_EMP10580_500_CTF_GN10_MTF_GT_split', dstype='clean'):
        super(MpiSintel, self).__init__(args,aug_params)
        flow_root = osp.join(root, split, 'flow')
        gt_root=osp.join(root, split, 'gt')
        image_root = osp.join(root, split, dstype)
        self.num_frames = args.num_frames
        # print("=============num_frames:", self.num_frames)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))

            for i in range(len(image_list)-self.num_frames+1):
                self.image_list += [ image_list[i:i+self.num_frames] ]

                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                flow_list=sorted(glob(osp.join(flow_root, scene, '*.flo')))
                gt_list = sorted(glob(osp.join(gt_root, scene, '*.png')))
                #print("+++++:",len(flow_list),len(gt_list))
                for i in range(len(image_list) - self.num_frames + 1):
                    self.flow_list += [ flow_list[i+int((self.num_frames-1)/2)] ]
                    self.gt_list += [gt_list[i+int((self.num_frames-1)/2)]]
                #print("+++++:", len(self.flow_list), len(self.gt_list))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        # print(args.num_frames)
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        # things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(args, aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(args, aug_params, split='training', dstype='final')
        train_dataset = 100*sintel_clean + 100*sintel_final
        # if TRAIN_DS == 'C+T+K+S+H':
        #     print("===========yes")
        #     kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
        #     hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
        #     train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things
        #
        # elif TRAIN_DS == 'C+T+K/S':
        #     print("============no")
        #     train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

