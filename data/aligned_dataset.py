import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        print("Building training dataset from ", self.dir_AB)
        self.AB_paths = sorted(make_dataset(self.dir_AB))

        if self.opt.coordconv == True:
            print("Adding coordconv layers to inputs...")
            assert self.opt.input_nc == 5, "You must set input_nc = 5 to accommodate i,j coordinates"
            self.i_coords, self.j_coords = np.meshgrid(range(self.opt.fineSize), range(self.opt.fineSize), indexing='ij')
        
        assert(opt.resize_or_crop == 'resize_and_crop')
        
        self.thresholds = [0.3118, 0.3137, 0.3196, 0.3059]
        
        #thresh_idx = AB_path.split('_')[0][-1]

        #transform_list = [transforms.ToTensor(),
         #                 transforms.Normalize((0.5, 0.5, 0.5),
          #                                     (0.5, 0.5, 0.5))]

        #self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        assert(self.opt.loadSize >= self.opt.fineSize)

        w, h = AB.size
        w2 = int(w / 2)

        A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A_tensor = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        B_tensor = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A_tensor)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B_tensor)
        
        #AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        #B_crop = AB.crop((256,0,512,256))
        #IF_mask =  IF_stain > self.thresholds[int(AB_path.split('_')[0][-1])-1]
        #IF_mask = 255*IF_mask.astype(dtype="uint8")
        #AB = self.transform(AB)

        #w_total = AB.size(2)
        #w = int(w_total / 2)
        #h = AB.size(1)
        #w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        #h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        #A = AB[:, h_offset:h_offset + self.opt.fineSize,
        #       w_offset:w_offset + self.opt.fineSize]
        #B = AB[:, h_offset:h_offset + self.opt.fineSize,
        #      w + w_offset:w + w_offset + self.opt.fineSize]
                
        #mask = torch.gt(B[0,:,:],-(1-self.thresholds[int(AB_path.split('_')[0][-1])-1]))

        if self.opt.coordconv == True:
            cc_layer = np.dstack((self.i_coords.astype("uint8"),self.j_coords.astype("uint8")))
            cc_layer = transforms.ToTensor()(cc_layer)
            cc_layer = transforms.Normalize((0.5, 0.5), (0.5, 0.5))(cc_layer)
            A = torch.cat((A,cc_layer))
            
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            thresh_idx = int(AB_path.split('_')[0][-1]) - 1
            thresh = self.thresholds[thresh_idx]
            mask_tmp = B_tensor[self.opt.stain_channel, ...]
            real_mask = mask_tmp.gt(thresh)
            tmp = B[self.opt.stain_channel, ...]
            B = tmp.unsqueeze(0)
            
            return {'A': A, 'B': B, 'real_mask': real_mask, 'thresh': thresh,
                'A_paths': AB_path, 'B_paths': AB_path}            
        else:
            
            return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
