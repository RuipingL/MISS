import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF 
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
import random
class CityScapes(Dataset):
    """
    num_classes: 25
    """
    CLASSES = ['Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain', 'Sky', 'Person', 'Rider', 'Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle']
    PALETTE = torch.tensor([[128, 64, 128],[244, 35, 232],[70, 70, 70],[102, 102, 156],[190, 153, 153],[153, 153, 153],[250, 170, 30],[220, 220, 0],[107, 142, 35],[152, 251, 152],[70, 130, 180],[220, 20, 60],[255, 0, 0],[0, 0, 142],[0, 0, 70],[0, 60, 100],[0, 80, 100],[0, 0, 230],[119, 11, 32]])
    
    def __init__(self, root: str = '/dataset/cityscapes', split: str = 'train', transform = None, modals = ['img'], miss = True, case = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.miss = miss
        self.files = sorted(glob.glob(os.path.join(*[root, 'leftImg8bit', split, '*', '*.png'])))
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} {case} images.")
        self.split = split

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        rgb = str(self.files[index])
        x1 = rgb.replace('/leftImg8bit', '/depth').replace('_leftImg8bit.png', '_depth.npy')
        lbl_path = rgb.replace('/leftImg8bit', '/gtFine').replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
        # rgb = rgb.replace('/img', '/missing').replace('_rgb', '_missing')

        sample = {}
        if self.split == 'train' and self.miss:
            # if self.miss:
            print('Missing')
            rgb_miss = random.getrandbits(1)
            depth_miss = random.getrandbits(1)
            if (rgb_miss == 1) and (depth_miss == 0):
                # print('RGB miss')
                # sample['img'] = torch.zeros_like(sample['img'])
                rgb = rgb.replace('/leftImg8bit', '/missing').replace('_leftImg8bit.png', '_missing.png')
                sample['img'] = io.read_image(rgb)[:3, ...]
                sample['depth'] = np.load(x1)
                sample['depth'] = torch.Tensor(sample['depth']).unsqueeze(0).expand(3, -1, -1)
            elif (depth_miss == 1) and (rgb_miss == 0):
                # print('Depth miss')
                x1 = x1.replace('/depth', '/missing').replace('_depth.npy', '_missing.png')
                sample['img'] = io.read_image(rgb)[:3, ...]
                sample['depth'] = io.read_image(x1)
            else:
                # print('No miss')
                sample['img'] = io.read_image(rgb)[:3, ...]
                sample['depth'] = np.load(x1)
                sample['depth'] = torch.Tensor(sample['depth']).unsqueeze(0).expand( 3, -1, -1)
        else:
            # print('No miss')
            sample['img'] = io.read_image(rgb)[:3, ...]
            sample['depth'] = np.load(x1)
            sample['depth'] = torch.Tensor(sample['depth']).unsqueeze(0).expand(3, -1, -1)

            # RGB missing
            # rgb = rgb.replace('/leftImg8bit', '/missing').replace('_leftImg8bit.png', '_missing.png')
            # sample['img'] = io.read_image(rgb)[:3, ...]
            # sample['depth'] = np.load(x1)
            # sample['depth'] = torch.Tensor(sample['depth']).unsqueeze(0).expand(3, -1, -1)
            
            ## Depth missing
            # sample['img'] = io.read_image(rgb)[:3, ...]
            # x1 = x1.replace('/depth', '/missing').replace('_depth.npy', '_missing.png')
            # sample['depth'] = io.read_image(x1)
        label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        # label[label==255] = 0
        # label -= 1
        sample['mask'] = label
        

                
        if self.transform:
            sample = self.transform(sample) # transformer color jitter on rgb
            
            
        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()
        sample = [sample[k] for k in self.modals]
        # if (rgb_miss == 1) and (depth_miss == 0):
        #     print(rgb)
        #     print('RGB missing', is_full_zero_tensor_gpu(sample[0]))
        return sample, label

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)

def is_full_zero_tensor_gpu(tensor):
    """
    Check if a GPU tensor is a full-zero tensor.

    Args:
        tensor: Input tensor on the GPU.

    Returns:
        True if all elements of the GPU tensor are zero, False otherwise.
    """
    # Move the tensor to CPU to perform the check
    tensor_cpu = tensor.cpu()

    # Check if all elements are zero on the CPU
    return torch.all(tensor_cpu == 0)
if __name__ == '__main__':


    # cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    # traintransform = get_train_augmentation((512, 1024), seg_fill=255)
    traintransform = get_val_augmentation((512, 1024))
    # for case in cases:

    trainset = CityScapes(transform=traintransform, split='val', modals=['img', 'depth'])
    trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=False, pin_memory=False)

    for i, (sample, lbl) in enumerate(trainloader):
        # print('Depth missing', is_full_zero_tensor_gpu(sample[1]))
        print(torch.unique(lbl))
        print(sample[0].shape)

