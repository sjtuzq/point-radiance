

import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

from modules.model import CoreModel
from modules.utils import device, mse2psnr, \
    grad_loss, safe_path, set_seed
from modules.config import config_parser
from dataloader.dataset import Data


class Trainder(object):
    def __init__(self, args):
        self.args = args
        self.dataname = args.dataname
        self.logpath = args.basedir
        self.outpath = safe_path(os.path.join(self.logpath, 'output'))
        self.weightpath = safe_path(os.path.join(self.logpath, 'weight'))
        self.imgpath = safe_path(os.path.join(self.outpath, 'images'))
        self.imgpath = safe_path(os.path.join(self.imgpath, '{}'.format(self.dataname)))
        self.logfile = os.path.join(self.outpath, 'log_{}.txt'.format(self.dataname))
        self.logfile = open(self.logfile, 'w')
        self.model = CoreModel(args).to(device)
        self.loss_fn = torch.nn.MSELoss()
        self.lr1, self.lr2  = args.lr1, args.lr2
        self.lrexp, self.lr_s = args.lrexp, args.lr_s
        self.set_optimizer(self.lr1, self.lr2)
        self.imagesgt = torch.tensor(self.model.imagesgt).float().to(device)
        self.masks = torch.tensor(self.model.masks).float().to(device)
        self.imagesgt_train = self.imagesgt
        self.imgout_path = safe_path(os.path.join(self.imgpath,
                        'v2_{:.3f}_{:.3f}'.format(args.data_r, args.splatting_r)))
        self.training_time = 0
        print(self.imgout_path)

    def set_onlybase(self):
        self.model.onlybase = True
        self.set_optimizer(3e-3,self.lr2)

    def remove_onlybase(self):
        self.model.onlybase = False
        self.set_optimizer(self.lr1,self.lr2)

    def set_optimizer(self, lr1=3e-3, lr2=8e-4):
        sh_list = [name for name, params in self.model.named_parameters() if 'sh' in name]
        sh_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in sh_list,
                                self.model.named_parameters()))))
        other_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in sh_list,
                                self.model.named_parameters()))))
        optimizer = torch.optim.Adam([
            {'params': sh_params, 'lr': lr1},
            {'params': other_params, 'lr': lr2}])
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lrexp, -1)
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        return None

    def train(self,epoch_n=30):
        self.logfile.write('-----------Stage Segmentation Line-----------')
        self.logfile.flush()
        max_psnr = 0.
        start_time = time.time()
        for epoch in range(epoch_n):
            loss_all, psnr_all = [], []
            ids = np.random.permutation(100)
            for id in tqdm(ids):
                images = self.model(id)
                loss = self.loss_fn(images[0], self.imagesgt_train[id])
                loss = loss + self.lr_s * grad_loss(images[0], self.imagesgt_train[id])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_all.append(loss)
                psnr_all.append(mse2psnr(loss))
            self.lr_scheduler.step()
            loss_e = torch.stack(loss_all).mean().item()
            psnr_e = torch.stack(psnr_all).mean().item()
            info = '-----train-----  epoch:{} loss:{:.3f} psnr:{:.3f}'.format(epoch, loss_e, psnr_e)
            print(info)
            self.logfile.write(info + '\n')
            self.logfile.flush()
            psnr_val = self.test(115, 138, False)
            if psnr_val > max_psnr:
                max_psnr = psnr_val
        self.training_time += time.time()-start_time
        torch.save(self.model.state_dict(), os.path.join(
                 self.weightpath,'model_{}.pth'.format(self.dataname)))

    def test(self, start=100, end=115, visual=False):
        plt.cla()
        plt.clf()
        with torch.no_grad():
            loss_all, psnr_all = [], []
            for id in (range(start, end)):
                images = self.model(id)
                loss = self.loss_fn(images[0], self.imagesgt[id])
                loss_all.append(loss)
                psnr_all.append(mse2psnr(loss))
                if visual:
                    pred = images[0, ..., :3].detach().cpu().data.numpy()
                    gt = self.imagesgt[id].detach().cpu().data.numpy()
                    # set background as white for visualization
                    mask = self.masks[id].cpu().data.numpy()
                    pred = pred*mask+1-mask
                    gt = gt*mask+1-mask
                    img_gt = np.concatenate((pred,gt),1)
                    img_gt = Image.fromarray((img_gt*255).astype(np.uint8))
                    img_gt.save(os.path.join(self.imgout_path,
                            'img_{}_{}_{:.2f}.png'.format(self.dataname, id, mse2psnr(loss).item())))
            loss_e = torch.stack(loss_all).mean().item()
            psnr_e = torch.stack(psnr_all).mean().item()
            info = '-----eval-----  loss:{:.3f} psnr:{:.3f}'.format(loss_e, psnr_e)
            print(info)
            self.logfile.write(info + '\n')
            self.logfile.flush()
            return psnr_e

    def get_fps_modelsize(self):
        start_time = time.time()
        for id in (range(0, 138)):
            images = self.model(id)
        end_time = time.time()
        fps = 138 / (end_time - start_time)
        model_path = os.path.join(
                 self.weightpath,'model_{}.pth'.format(self.dataname))
        model_size = os.path.getsize(model_path)
        model_size = model_size / float(1024 * 1024)
        model_size = round(model_size, 2)
        return fps,model_size


def solve(args):
    trainer = Trainder(args)
    trainer.set_onlybase()
    trainer.train(epoch_n=20)
    trainer.remove_onlybase()
    trainer.train()
    for i in range(args.refine_n):
        trainer.model.remove_out()
        trainer.model.repeat_pts()
        trainer.set_optimizer(args.lr1, args.lr2)
        trainer.train()
    trainer.logfile.write('Total Training Time: '
                  '{:.2f}s\n'.format(trainer.training_time))
    trainer.logfile.flush()
    psnr_e = trainer.test(115, 138, True)
    fps,model_size = trainer.get_fps_modelsize()
    print('Training time: {:.2f} s'.format(trainer.training_time))
    print('Rendering quality: {:.2f} dB'.format(psnr_e))
    print('Rendering speed: {:.2f} fps'.format(fps))
    print('Model size: {:.2f} MB'.format(model_size))


if __name__ == '__main__':
    set_seed(0)
    parser = config_parser()
    args = parser.parse_args()
    dataset = Data(args)
    args.memitem = dataset.genpc()
    solve(args)
