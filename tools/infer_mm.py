import torch
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
import torchvision.transforms.functional as TF 
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text
import glob
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette), cfg['DATASET']['MODALS'])
        msg = self.model.load_state_dict(torch.load(cfg['EVAL']['MODEL_PATH'], map_location='cpu'))
        print(msg)
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline_img = T.Compose([
            T.Resize(self.size),
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])
        self.tf_pipeline_modal = T.Compose([
            T.Resize(self.size),
            T.Lambda(lambda x: x / 255),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        seg_image = self.palette[seg_map].squeeze()
        if overlay: 
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        image = seg_image.to(torch.uint8)
        pil_image = Image.fromarray(image.numpy())
        print(pil_image.size)
        return pil_image

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)
    
    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def predict(self, img_fname: str, overlay: bool) -> Tensor:
        if cfg['DATASET']['NAME'] == 'DELIVER':
            x1 = img_fname.replace('/img', '/depth').replace('_rgb', '_depth')
            x2 = img_fname.replace('/img', '/lidar').replace('_rgb', '_lidar')
            x3 = img_fname.replace('/img', '/event').replace('_rgb', '_event')
            lbl_path = img_fname.replace('/img', '/semantic').replace('_rgb', '_semantic')
            
            # img_fname = img_fname.replace('/img', '/missing').replace('_rgb', '_missing')
            x1 = img_fname.replace('/img', '/missing').replace('_rgb', '_missing')
            
            image = io.read_image(img_fname)[:3, ...]
            img = self.tf_pipeline_img(image).to(self.device)
            # --- modals
            x1 = self._open_img(x1)
            x1 = self.tf_pipeline_modal(x1).to(self.device)
            x2 = self._open_img(x2)
            x2 = self.tf_pipeline_modal(x2).to(self.device)
            x3 = self._open_img(x3)
            x3 = self.tf_pipeline_modal(x3).to(self.device)
            label = io.read_image(lbl_path)[0,...].unsqueeze(0)
            label[label==255] = 0
            label -= 1
        elif cfg['DATASET']['NAME'] == 'CityScapes':
            miss = 'depth_miss'
            # miss = 'rgb_miss'
            # miss = 'rgb_d'
            # x1 = img_fname.replace('/leftImg8bit', '/depth').replace('_leftImg8bit.png', '_depth.npy')
            # if miss=='rgb_d' or miss == 'rgb_miss':
            x1 = img_fname.replace('/leftImg8bit', '/depth').replace('_leftImg8bit.png', '_depth.npy')
            x1 = np.load(x1)
            x1 = torch.from_numpy(x1).float().unsqueeze(0).expand( 3, -1, -1)
            x1 = torch.zeros_like(x1)
            x1 = self.tf_pipeline_modal(x1).to(self.device)
            # elif miss == 'depth_miss':
            #     print('Depth miss')
            #     x1 = img_fname.replace('/leftImg8bit', '/missing').replace('_leftImg8bit.png', '_missing.png')
            #     # x1 = x1.replace('/depth', '/missing').replace('_depth.npy', '_missing.png')
            #     x1 = io.read_image(x1)
            #     x1 = self.tf_pipeline_img(x1).to(self.device)
            # else:
            #     raise NotImplementedError()
            
            # if miss == 'rgb_miss':
            #     img_fname = img_fname.replace('/leftImg8bit', '/missing').replace('_leftImg8bit.png', '_missing.png')
            image = io.read_image(img_fname)[:3, ...]
            img = self.tf_pipeline_img(image).to(self.device)

            
            lbl_path = img_fname.replace('/leftImg8bit', '/gtFine').replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
            label = io.read_image(lbl_path)[0,...].unsqueeze(0)
            
        # sample = [img, x1, x2, x3][:len(modals)]
        sample = [img, x1]

        seg_map = self.model_forward(sample)
        seg_map = self.postprocess(image, seg_map, overlay)
        return seg_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/DELIVER.yaml')
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres', None]
    # cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    # cases = ['lidarjitter']
    cases = None
    

    modals = cfg['DATASET']['MODALS']
    test_file = Path(cfg['TEST']['FILE'])
    subfolder = cfg['TEST']['MODEL_PATH'].split('/')[-2]
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    # print(f"Model {cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}")
    # print(f"Model {cfg['DATASET']['NAME']}")

    modals_name = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    save_dir = Path(cfg['SAVE_DIR']) / subfolder/ 'test_depthmiss_results' / (cfg['DATASET']['NAME']+'_'+cfg['MODEL']['BACKBONE']+'_'+modals_name)
    os.makedirs(save_dir, exist_ok=True)
    semseg = SemSeg(cfg)

    if test_file.is_file():
        segmap = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
        segmap.save(save_dir / f"{str(test_file.stem)}.png")
    else:
        if cfg['DATASET']['NAME'] == 'DELIVER':
            files = sorted(glob.glob(os.path.join(*[str(test_file), 'img', '*', 'val', '*', '*.png']))) # --- Deliver
        # elif cfg['DATASET']['NAME'] == 'KITTI360':
        #     source = os.path.join(test_file, 'val.txt')
        #     files = []
        #     with open(source) as f:
        #         files_ = f.readlines()
        #     for item in files_:
        #         file_name = item.strip()
        #         if ' ' in file_name:
        #             # --- KITTI-360
        #             file_name = os.path.join(*[str(test_file), file_name.split(' ')[0]])
        #         files.append(file_name)
        elif cfg['DATASET']['NAME'] == 'CityScapes':
            files = sorted(glob.glob(os.path.join(*[str(test_file), 'leftImg8bit', 'val', '*', '*.png'])))
            # print(files)
        else:
            raise NotImplementedError()

        for file in files:
            # print(file)
            # if not '2013_05_28_drive_0000_sync' in file:
            #     continue
            segmap = semseg.predict(file, cfg['TEST']['OVERLAY'])
            file = file.split('/')[-1]
            save_path = os.path.join(str(save_dir),file)
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            segmap.save(save_path)
