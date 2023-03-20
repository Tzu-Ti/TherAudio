import argparse
from tqdm import tqdm
import visdom
import os

from multiprocessing import cpu_count

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.optim as optim

from utils import Visdom

from models.Identity import IdentityAE

from dataset import IdentityDataset

from torch_ema import ExponentialMovingAverage
from copy import deepcopy

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # training setting
    parser.add_argument('--model_name', default='test')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    # I/O setting
    parser.add_argument('--saving_folder', default='ckpts')
    # 
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpus', type=int, default=4)
    # Visdom setting
    parser.add_argument('--port', type=int, default=1203)
    parser.add_argument('--env', default="test")
    parser.add_argument('--visual_loss_step', type=int, default=10)
    parser.add_argument('--visual_output_step', type=int, default=50)
    # Mode
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--resume', action="store_true")
    
    return parser.parse_args()

class Model_factory():
    def __init__(self, args, vis):
        self.args = args
        self.vis = vis
        self.device = args.device
        
        trainDataset = IdentityDataset(augment=True, train=True)
        self.trainDataloader = DataLoader(dataset=trainDataset,
                                          batch_size=args.batch_size,
                                          shuffle=True, 
                                          num_workers=cpu_count())
        testDataset = IdentityDataset(augment=False, train=False)
        self.testDataloader = DataLoader(dataset=testDataset,
                                         batch_size=args.batch_size,
                                         shuffle=False, 
                                         num_workers=cpu_count())
        
        self.AE = IdentityAE(input_size=[384, 512], hidden_size=256, num_layers=2, heads=8, dropout=0.1)
        self.AE = DataParallel(self.AE, device_ids=[i for i in range(self.args.gpus)]).to(self.device)
        self.ema = ExponentialMovingAverage(self.AE.parameters(), decay=0.995)
        
        self.optimizer = optim.AdamW(self.AE.parameters(), lr=args.lr)
        self.rec_criterion = nn.L1Loss()
        
        self.saving_folder = os.path.join(args.saving_folder, args.model_name)
        
    def draw_visualization(self, rec, img, cols=4):
        rec = rec[:cols]
        img = img[:cols]
        
        return torch.cat([rec, img], dim=0)
    
    def train(self, e):
        self.AE.train()
        l = len(self.trainDataloader)
        pbar = tqdm(self.trainDataloader)
        for i, img in enumerate(pbar):
            self.img = img.to(self.device)

            rec = self.AE(self.img)
            loss = self.rec_criterion(self.img, rec)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema.update()
            
            pbar.set_description("Loss: {:.05f}".format(loss.item()))
            
            if i % self.args.visual_loss_step == 0:
                self.vis.Line(loss_type="L1", win='Loss', loss=loss.item(), step=l*(e-1)+i)
            if i % self.args.visual_output_step == 0:
                vis_img = self.draw_visualization(rec=rec, img=self.img)
                self.vis.Images(images=vis_img, win='Current img', ncol=4, unormalize=True)
                
    def validate(self, e):
        img = next(iter(self.testDataloader)).to(self.device)
        with self.ema.average_parameters():
            rec = self.AE(img)
        vis_img = self.draw_visualization(rec=rec, img=img)
        self.vis.Images(images=vis_img, win='Epoch-{}'.format(e), ncol=4, unormalize=True)
            
    def save(self, e):
        print("Saving Model...")
        if not os.path.isdir(self.saving_folder): os.makedirs(self.saving_folder)
        ckpt = {
            "parameter": self.AE.state_dict(),
            "epoch": e,
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
        }
        torch.save(ckpt, '{}/checkpoint.ckpt'.format(self.saving_folder))
        
        ema_model = deepcopy(self.AE)
        self.ema.copy_to(ema_model.parameters())
        torch.save(ema_model, '{}/AE.pt'.format(self.saving_folder))
        
    def load(self):
        print("Loading Model...")
        ckpt = torch.load('{}/checkpoint.ckpt'.format(self.saving_folder))
        self.AE.load_state_dict(ckpt['parameter'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.ema.load_state_dict(ckpt['ema'])
        
        epoch = ckpt['epoch']
        return epoch
            
def main():
    args = parse()
    vis = Visdom(args.env, args.port)
    Model = Model_factory(args, vis)
    
    start_epoch = Model.load()+1 if args.resume else 1
    if args.train:
        for e in range(start_epoch, args.epochs+1):
            print("Epoch: {}".format(e))
            Model.train(e)
            
            if e % 5 == 0:
                print("Validating...")
                Model.validate(e)
                Model.save(e)
                
if __name__ == '__main__':
    main()
        