import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
from random import randint, seed
from torch import cat as tc
import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries

class WSIUnalignedDataset(BaseDataset):
    
    def initialize(self, opt):

        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        print('dir_A: ', self.dir_A)
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        print('dir_B: ', self.dir_B)
        self.A_paths = make_dataset(self.dir_A)
        print('A_paths: ', self.A_paths)
        self.B_paths = make_dataset(self.dir_B)
        print('B_paths: ', self.B_paths)
        
        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        assert self.A_size == self.B_size, "HE/IF image mismatch. HE and IF images must be paired."
        
        if self.opt.coordconv == True:
            print("Adding coordconv layers to inputs...")
            assert self.opt.input_nc == 5, "You must set input_nc = 5 to accommodate i,j coordinates"
            self.i_coords, self.j_coords = np.meshgrid(range(self.opt.fineSize), range(self.opt.fineSize), indexing='ij')
        
        tpi = transforms.ToPILImage()
        Image.MAX_IMAGE_PIXELS = None
        self.HE_WSIs = []
        self.IF_WSIs = []
        raw_weights = []
        
        for HE_WSI, IF_WSI in zip(self.A_paths,self.B_paths):
            
            print('Adding HE img {} and IF img {} to dataset...'.format(HE_WSI.split('/')[-1],IF_WSI.split('/')[-1]))
            
            HE_img = Image.open(HE_WSI).convert('RGB')
            IF_img = Image.open(IF_WSI).convert('RGB')
            
            assert HE_img.size == IF_img.size, "Dimension mismatch error: HE/IF images must be of same dimension."
            
            if self.opt.resize_factor != 1:
                new_dims = tuple(int(np.floor(self.opt.resize_factor * i)) for i in HE_img.size)
                print(f"Downsampling images from {HE_img.size} to {new_dims}...")
                HE_img = HE_img.resize(new_dims,Image.LANCZOS)
                IF_img = IF_img.resize(new_dims,Image.LANCZOS)
                self.HE_WSIs.append(HE_img)
            else:
                self.HE_WSIs.append(HE_img)
            
            # Compute weight for each image based on its number of non-black-or-white pixels.
            # Weights will be used during batch generation to prevent overfitting on smaller images,
            # i.e. the larger the WSI, the greater the probability of drawing a sample from that WSI.
            HE_arr = np.array(HE_img)
            curr_weight = ((HE_arr!=0).any(axis=2)&(HE_arr!=255).any(axis=2)).sum()
            raw_weights.append(curr_weight)
            
            red, green, blue = IF_img.split() # TODO: add option to select IF color channel
            IF_stain = np.array(green,dtype="uint8")
            
            if self.opt.output_nc == 1:
                IF_img = tpi(IF_stain[:,:,np.newaxis])
                self.IF_WSIs.append(IF_img)
            
            elif self.opt.output_nc == 2:
                IF_stain_thresh = threshold_otsu(IF_stain)
                IF_mask =  IF_stain > IF_stain_thresh
                IF_mask = 255*IF_mask.astype(dtype="uint8")
                IF_stain.shape
                IF_mask.shape
                IF_img = np.dstack((IF_stain,IF_mask))
                self.IF_WSIs.append(IF_img)
            
            elif self.opt.output_nc == 3:
                IF_stain_thresh = threshold_otsu(IF_stain)
                IF_mask =  IF_stain > IF_stain_thresh
                IF_mask = 255*IF_mask.astype(dtype="uint8")
                IF_bounds = find_boundaries(IF_mask)
                IF_bounds = 255*IF_bounds.astype(dtype="uint8")
                IF_img = tpi(np.dstack((IF_stain,IF_mask,IF_bounds)))
                self.IF_WSIs.append(IF_img)
        
        self.weights = [weight / sum(raw_weights) for weight in raw_weights]
        for img, weight in zip(self.A_paths,self.weights):
            print(img.split("/")[-1]+": ", weight)
        
        self.in_transform, self.out_transform = get_transform(opt)
        
    def __getitem__(self, index):
        
        rand_crop = transforms.RandomCrop(self.opt.fineSize)
        
        while True:
            
            A_path = self.A_paths[index]
            B_path = self.B_paths[index]
            rand_seed = randint(0,2**32)
            seed(rand_seed)
            A_rand = rand_crop(self.HE_WSIs[index])
            A_arr = np.array(A_rand)
            if ((A_arr==255).all(axis=2).sum() + (A_arr==0).all(axis=2).sum()) == 0:
                break
        
        if self.opt.coordconv == True:
            A_rand = np.dstack((A_arr,self.i_coords.astype("uint8"),self.j_coords.astype("uint8")))
            
        A = self.in_transform(A_rand)

        seed(rand_seed)
        B = self.out_transform(rand_crop(self.IF_WSIs[index]))
        
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'WSIUnalignedDataset'
