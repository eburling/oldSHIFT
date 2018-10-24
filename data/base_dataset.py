import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
#    elif opt.resize_or_crop == 'crop':
#        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
#    elif opt.contrast == True:
#        transform_list.append(transforms.functional.adjust_contrast(
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    in_transform_list = [transforms.ToTensor()]
    out_transform_list = [transforms.ToTensor()]
    
    if opt.coordconv == True:
        in_transform_list.append(transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5, 0.5)))
    else:
        in_transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    if opt.output_nc == 3:
        out_transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif opt.output_nc == 2:
        out_transform_list.append(transforms.Normalize((0.5, 0.5), (0.5, 0.5)))
    elif opt.output_nc == 1:
        out_transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    print(in_transform_list,out_transform_list)                  
    return transforms.Compose(in_transform_list), transforms.Compose(out_transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
