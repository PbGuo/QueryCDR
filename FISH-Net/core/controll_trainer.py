import os
import time
import glob
import importlib

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from core.controll_dataset import Dataset
from core.utils import set_seed, set_device, Progbar, postprocess
from core.loss import AdversarialLoss, PerceptualLoss, StyleLoss, VGG19


class Trainer():
    def __init__(self, config, debug=False):
        self.config = config
        self.epoch = 0
        self.last_epoch = 0
        self.iteration = 0
        if debug:
            self.config['trainer']['save_freq'] = 5
            self.config['trainer']['valid_freq'] = 5
            self.config['trainer']['save_epoch'] = 1
            self.config['trainer']['valid_epoch'] = 1

        # setup data set and data loader
        self.train_dataset = Dataset(config['data_loader'], debug=debug, split='train')
        worker_init_fn = partial(set_seed, base=config['seed'])
        self.train_sampler = None
        if config['distributed']:
            self.train_sampler = DistributedSampler(self.train_dataset,
                                                    num_replicas=config['world_size'], rank=config['global_rank'])
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=config['trainer']['batch_size'] // config['world_size'],
                                       shuffle=(self.train_sampler is None),
                                       # num_workers=config['trainer']['num_workers'],
                                       pin_memory=True, sampler=self.train_sampler, worker_init_fn=worker_init_fn)

        # set up losses and metrics
        self.adversarial_loss = set_device(AdversarialLoss(type=self.config['losses']['gan_type']))
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()

        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            self.dis_writer = SummaryWriter(os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(os.path.join(config['save_dir'], 'gen'))
        self.train_args = self.config['trainer']

        net = importlib.import_module('model.' + config['model_name'])
        self.netG = set_device(net.InpaintGenerator())
        self.netD = set_device(net.Discriminator(in_channels=3, use_sigmoid=config['losses']['gan_type'] != 'hinge'))
        self.optimG = torch.optim.Adam(self.netG.parameters(), lr=config['trainer']['lr'],
                                       betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.optimD = torch.optim.Adam(self.netD.parameters(), lr=config['trainer']['lr'] * config['trainer']['d2glr'],
                                       betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.load()
        if config['distributed']:
            self.netG = DDP(self.netG, device_ids=[config['global_rank']], output_device=config['global_rank'],
                            broadcast_buffers=True, find_unused_parameters=True)
            self.netD = DDP(self.netD, device_ids=[config['global_rank']], output_device=config['global_rank'],
                            broadcast_buffers=True, find_unused_parameters=True)

    # get current learning rate
    def get_lr(self, type='G'):
        if type == 'G':
            return self.optimG.param_groups[0]['lr']
        return self.optimD.param_groups[0]['lr']

    # learning rate scheduler, step
    def adjust_learning_rate(self):
        decay = 0.1 ** (min(self.iteration, self.config['trainer']['niter_steady']) // self.config['trainer']['niter'])
        new_lr = self.config['trainer']['lr'] * decay
        if new_lr != self.get_lr():
            for param_group in self.optimG.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimD.param_groups:
                param_group['lr'] = new_lr

    # load netG and netD
    def load(self):
        model_path = self.config['save_dir']

        if self.config['loadfrom'] == True:
            model_path = self.config['save_dir']
            print('Loading model from {}...'.format(self.config['save_dir']))
            latest_epoch = self.config['loadnum']
            gen_path = os.path.join(model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
            dis_path = os.path.join(model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_path = os.path.join(model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(gen_path))
            data = torch.load(gen_path, map_location=lambda storage, loc: set_device(storage))
            self.netG.load_state_dict(data['netG'],  strict=self.config['loadstrict'])
            data = torch.load(dis_path, map_location=lambda storage, loc: set_device(storage))
            self.netD.load_state_dict(data['netD'],  strict=self.config['loadstrict'])
            # data = torch.load(opt_path, map_location=lambda storage, loc: set_device(storage))
            # self.optimG.load_state_dict(data['optimG'])
            # self.optimD.load_state_dict(data['optimD'])
            # self.epoch = data['epoch']
            # self.iteration = data['iteration']

        else:
            print("Load auto!")
            if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
                latest_epoch = open(os.path.join(model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
            else:
                ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(os.path.join(model_path, '*.pth'))]
                ckpts.sort()
                latest_epoch = ckpts[-1] if len(ckpts) > 0 else None
            if latest_epoch is not None:
                gen_path = os.path.join(model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
                dis_path = os.path.join(model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
                opt_path = os.path.join(model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
                if self.config['global_rank'] == 0:
                    print('Loading model from {}...'.format(gen_path))
                data = torch.load(gen_path, map_location=lambda storage, loc: set_device(storage))
                self.netG.load_state_dict(data['netG'])
                data = torch.load(dis_path, map_location=lambda storage, loc: set_device(storage))
                self.netD.load_state_dict(data['netD'])
                data = torch.load(opt_path, map_location=lambda storage, loc: set_device(storage))
                self.optimG.load_state_dict(data['optimG'])
                self.optimD.load_state_dict(data['optimD'])
                self.epoch = data['epoch']
                self.iteration = data['iteration']
            else:
                if self.config['global_rank'] == 0:
                    print('Warnning: There is no trained model found. An initialized model will be used.')
        
        if self.config['finetune'] == True:
            print("Finetune!")
            print("copying weights...")
            self.netG.conv_column.copy_weight()
            self.netG.flow_column.copy_weight()
            self.netD.copy_weight()


    # save parameters every eval_epoch
    def save(self, it):
        if self.config['global_rank'] == 0:
            gen_path = os.path.join(self.config['true_save_dir'], 'gen_{}.pth'.format(str(it).zfill(5)))
            dis_path = os.path.join(self.config['true_save_dir'], 'dis_{}.pth'.format(str(it).zfill(5)))
            opt_path = os.path.join(self.config['true_save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))
            print('\nsaving model to {} ...'.format(gen_path))
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG, netD = self.netG.module, self.netD.module
            else:
                netG, netD = self.netG, self.netD
            torch.save({'netG': netG.state_dict()}, gen_path)
            torch.save({'netD': netD.state_dict()}, dis_path)
            torch.save({'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict()}, opt_path)
            os.system('echo {} > {}'.format(str(it).zfill(5), os.path.join(self.config['true_save_dir'], 'latest.ckpt')))

    def add_summary(self, writer, name, val):
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name] / 100, self.iteration)
            self.summary[name] = 0

    # process input and calculate loss every training epoch
    def _train_epoch(self):
        progbar = Progbar(len(self.train_dataset), width=20, stateful_metrics=['epoch', 'iter'])
        mae = 0
        #for images, gt, _, _ in self.train_loader:
        for pairs in self.train_loader:
            self.iteration += 1
            self.add_summary(self.dis_writer, 'lr/dis_lr', self.get_lr(type='D'))
            self.add_summary(self.gen_writer, 'lr/gen_lr', self.get_lr(type='G'))
            self.adjust_learning_rate()
            end = time.time()

            sigma_it=[1, 0.2, 0.4, 0.6, 0.8, 2, 4, 6, 8]
            for iteration, (images, gt) in enumerate(pairs):

                gen_loss = 0
                dis_loss = 0
                
                images, gt = set_device([images, gt])
                it_sig = iteration
                feats, pred_img = self.netG(images, it_sig)
                comp_img = pred_img

                # image discriminator loss
                dis_real_feat = self.netD(gt, it_sig)
                dis_fake_feat = self.netD(comp_img.detach(), it_sig)
                dis_real_loss = self.adversarial_loss(dis_real_feat, True, True)
                dis_fake_loss = self.adversarial_loss(dis_fake_feat, False, True)
                dis_loss += (dis_real_loss + dis_fake_loss) / 2
                self.add_summary(self.dis_writer, 'loss/dis_fake_loss', dis_fake_loss.item())

                self.optimD.zero_grad()
                dis_loss.backward()
                self.optimD.step()

                # generator adversarial loss
                gen_fake_feat = self.netD(comp_img, it_sig)
                gen_fake_loss = self.adversarial_loss(gen_fake_feat, True, False)
                gen_loss += gen_fake_loss * self.config['losses']['adversarial_weight']
                self.add_summary(self.gen_writer, 'loss/gen_fake_loss', gen_fake_loss.item())

                # generator l1 loss
                L1_loss = self.l1_loss(comp_img, gt)
                gen_loss += L1_loss * self.config['losses']['l1_weight']
                self.add_summary(self.gen_writer, 'loss/L1_loss', L1_loss.item())

                # perceptual loss
                gen_content_loss = self.perceptual_loss(comp_img, gt)
                gen_loss += gen_content_loss * self.config['losses']['content_loss_weight']
                self.add_summary(self.gen_writer, 'loss/content_loss', gen_content_loss.item())

                # style loss
                gen_style_loss = self.style_loss(comp_img, gt)
                gen_loss += gen_style_loss * self.config['losses']['style_loss_weight']
                self.add_summary(self.gen_writer, 'loss/style_loss', gen_style_loss.item())

                if feats is not None:
                    pyramid_loss = 0
                    for _, f in enumerate(feats):
                        pyramid_loss += self.l1_loss(f, F.interpolate(gt, size=f.size()[2:4], mode='bilinear',
                                                                    align_corners=True))
                    gen_loss += pyramid_loss * self.config['losses']['pyramid_weight']
                    self.add_summary(self.gen_writer, 'loss/pyramid_loss', pyramid_loss.item())
                else:
                    pyramid_loss = torch.tensor(0)

                self.optimG.zero_grad()
                gen_loss.backward()
                self.optimG.step()




            new_mae = (torch.mean(torch.abs(gt - pred_img))).item()
            mae = new_mae
            logs = [("epoch", self.epoch), ("iter", self.iteration), ("lr", self.get_lr()),
                    ('mae', mae), ('gen_loss', gen_fake_loss.item()), ('L1_loss', L1_loss.item()),
                    ('pyramid_loss', pyramid_loss.item()),
                    ('content_loss', gen_content_loss.item()),
                    ('style_loss', gen_style_loss.item())]

            if self.config['global_rank'] == 0:
                progbar.add(len(images) * self.config['world_size'], values=logs \
                    if self.train_args['verbosity'] else [x for x in logs if not x[0].startswith('l_')])

            # # saving and evaluating
            # if self.iteration % self.train_args['save_freq'] == 0:
            #     self.save(int(self.iteration // self.train_args['save_freq']))
            # if self.iteration > self.config['trainer']['iterations']:
            #     break
            # saving and evaluating
            if self.epoch >= self.last_epoch + self.config['trainer']['save_epoch']:
                self.last_epoch = self.epoch
                self.save(int(self.epoch))
            if self.epoch > self.config['trainer']['epoch_num']:
                break

    def train(self):
        while True:
            self.epoch += 1
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)
            self._train_epoch()
            # if self.iteration > self.config['trainer']['iterations']:
            #     break
            if self.epoch > self.config['trainer']['epoch_num']:
                break
        print('\nEnd training....')

