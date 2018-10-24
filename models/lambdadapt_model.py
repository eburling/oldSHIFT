import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

def dice_loss(inp, target):
    smooth = 1.

    iflat = inp.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        #self.thresholds = [0.3118, 0.3137, 0.3196, 0.3059]
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_mask = input['real_mask'].to(self.device)
        self.thresh = input['thresh']
        #self.input_A.resize_(input_A.size()).copy_(input_A)
        #self.input_B.resize_(input_B.size()).copy_(input_B)
        #self.mask.resize_(mask.size()).copy_(mask)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        #self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)
        #self.real_B = Variable(self.input_B)
        self.fake_mask = transforms.Normalize(mean=[-1,-1,-1],std=[2,2,2])(self.fake_B.clone()).gt(self.thresh).float()

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # Second, G(A) = B
        if self.opt.output_nc > 1:
            #TODO: functionalize prevalence

            if self.opt.lambdadapt == True:
                prevalence = (1 / ((self.mask.sum().item() / (self.opt.fineSize ** 2)) + 0.01))
                #tensor_thresh = -(1-self.thresholds[int(AB_path.split('_')[0][-1])-1])
                
                #prevalence = (1 /
                 #             ((torch.sum(torch.gt(self.input_B[:,1,:,:],tensor_thresh)).item() / 
                  #              (self.opt.fineSize * self.opt.fineSize)) + 0.01))
                
                self.loss_G_L1 = self.criterionL1(self.fake_B[:,0,:,:], self.real_B[:,0,:,:]) * self.opt.lambda_A * prevalence
                
                if self.opt.loss_mask_L1 == True:
                    real_mask = torch.gt(self.real_B[0,:,:],tensor_thresh)
                    fake_mask = torch.gt(self.fake_B[0,:,:],tensor_thresh)
                    self.loss_mask = self.criterionL1() ## need to find path
                
                if self.opt.dice_loss == True:
                    
                    self.loss_dice = dice_loss(self.real_B[:,1,:,:], self.fake_B[:,1,:,:]) * prevalence
            
            else:
                
                self.loss_G_L1 = self.criterionL1(self.fake_B[:,0,:,:], self.real_B[:,0,:,:]) * self.opt.lambda_A
            
                if self.opt.dice_loss == True:
                    self.loss_dice = dice_loss(self.fake_B[:,1,:,:], self.real_B[:,1,:,:])

        else:
            
            if self.opt.lambdadapt == True:
                
                #tensor_thresh = -(1-self.thresholds[int(AB_path.split('_')[0][-1])-1])
                #real_mask = torch.gt(self.input_B[:,0,:,:],tensor_thresh)
                prevalence = (1 / ((self.real_mask.sum().item() / (self.opt.fineSize ** 2)) + 0.01)) 
                
                self.loss_G_L1 = self.criterionL1(self.fake_B[:,0,:,:], self.real_B[:,0,:,:]) * self.opt.lambda_A * prevalence
            
            else:
                
                self.loss_G_L1 = self.criterionL1(self.fake_B[:,0,:,:], self.real_B[:,0,:,:]) * self.opt.lambda_A
                
            if self.opt.dice_loss == True:
                self.loss_dice = dice_loss(self.fake_mask, self.real_mask)
                #self.loss_dice = dice_loss(torch.gt(self.fake_B[:,0,:,:],-0.8).float(), torch.gt(self.real_B[:,0,:,:],-0.8).float())
        
        if self.opt.dice_loss == True:
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_dice
        else:
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        
        self.set_requires_grad(self.netD, True)#
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        
        self.set_requires_grad(self.netD, False)#
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        losses = OrderedDict([('G_GAN', self.loss_G_GAN.data.item()),
                            ('G_L1', self.loss_G_L1.data.item()),
                            ('D_real', self.loss_D_real.data.item()),
                            ('D_fake', self.loss_D_fake.data.item())
                            ])
        
        if self.opt.dice_loss:
            losses['G_dice'] = self.loss_dice.data.item()
        
        return losses
            

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A[:,:,0:3]), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
