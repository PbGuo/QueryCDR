# -*- coding: utf-8 -*-

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from PIL import Image
import math
import os
import argparse
import importlib
import datetime
import json

### My libs
from core.utils import set_device, set_seed
from core.utils import postprocess
from core.controll_dataset import Dataset

parser = argparse.ArgumentParser(description="MGP")
parser.add_argument("-c", "--config", default='configs/querycdr.json', type=str, required=False)
parser.add_argument("-l", "--level",  type=int, required=False)
parser.add_argument("-m", "--mask", default='square', type=str)
parser.add_argument("-s", "--size", default=256, type=int)
parser.add_argument("-p", "--port", type=str, default="23451")
parser.add_argument("-b", "--batch", type=int, default="4")
parser.add_argument("-w", "--weight", type=str, default="latest")
args = parser.parse_args()

BATCH_SIZE = args.batch

def main_worker(gpu, ngpus_per_node, config):
  torch.cuda.set_device(gpu)
  set_seed(config['seed'])

  # Model and version
  net = importlib.import_module('model.'+config['model_name'])
  model = set_device(net.InpaintGenerator())
  if args.weight == "latest":
    latest_epoch = open(os.path.join(config['save_dir'], 'latest.ckpt'), 'r').read().splitlines()[-1]
  else:
    latest_epoch = args.weight
  path = os.path.join(config['save_dir'], 'gen_{}.pth'.format(latest_epoch))
  data = torch.load(path, map_location = lambda storage, loc: set_device(storage)) 
  model.load_state_dict(data['netG'])
  model.eval()

  # prepare dataset
  dataset = Dataset(config['data_loader'], debug=False, split='test', level=args.level)
  print("Dataset size: ", len(dataset))
  step = math.ceil(len(dataset) / ngpus_per_node)
  dataset.set_subset(gpu*step, min(gpu*step+step, len(dataset)))
  dataloader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=False, num_workers=config['trainer']['num_workers'], pin_memory=True)


  #path = os.path.join(config['save_dir'], 'results_{}_level_{}'.format(str(latest_epoch).zfill(5), str(args.level).zfill(2)))
  path = os.path.join(config['save_dir'], 'results_{}_level_{}'.format(str(latest_epoch).zfill(5), str(args.level).zfill(2)))
  os.makedirs(path, exist_ok=True)
  # iteration through datasets
  
  sig_dit={0:1, 1:0.2, 2:0.4, 3:0.6, 4:0.8, 5:2, 6:4, 7:6, 8:8}
  for idx, pairs in enumerate(dataloader):
    for it_sig, (images, gt) in enumerate(pairs):
        print('[{}] GPU{} {}/{}: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            gpu, idx, len(dataloader), idx))
        images, gt = set_device([images, gt])
        with torch.no_grad():
            _, output = model(images, it_sig)
        orig_imgs = postprocess(images)
        gt_imgs = postprocess(gt)
        pred_imgs = postprocess(output)
        for i in range(len(orig_imgs)):
            Image.fromarray(pred_imgs[i]).save(os.path.join(path, '{}_{}_{}_{}_pred.png'.format(gpu,idx,i,sig_dit[it_sig])))
            Image.fromarray(orig_imgs[i]).save(os.path.join(path, '{}_{}_{}_{}_orig.png'.format(gpu,idx,i,sig_dit[it_sig])))
            Image.fromarray(gt_imgs[i]).save(os.path.join(path, '{}_{}_{}_{}_gt.png'.format(gpu,idx,i,sig_dit[it_sig])))

  print('Finish in {}'.format(path))



if __name__ == '__main__':
  ngpus_per_node = torch.cuda.device_count()
  config = json.load(open(args.config))
  if args.mask is not None:
    config['data_loader']['mask'] = args.mask
  if args.size is not None:
    config['data_loader']['w'] = config['data_loader']['h'] = args.size
  # config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}'.format(config['model_name'], 
  #   config['data_loader']['name'], config['data_loader']['mask'], config['data_loader']['w']))
  config['save_dir'] = os.path.join(config['save_dir'])
  print(config['model_name'])
  print("Loading Folder ",config['save_dir'])
  print('using {} GPUs for testing ... '.format(ngpus_per_node))
  # setup distributed parallel training environments
  ngpus_per_node = torch.cuda.device_count()
  config['world_size'] = ngpus_per_node
  config['init_method'] = 'tcp://127.0.0.1:'+ args.port 
  config['distributed'] = True
  mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
 
