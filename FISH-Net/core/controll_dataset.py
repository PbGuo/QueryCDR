import random
from random import shuffle
import os 
import math 
import numpy as np 
from PIL import Image, ImageFilter
from glob import glob

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


from core.utils import ZipReader


class Dataset(torch.utils.data.Dataset):
  def __init__(self, data_args, debug=False, split='train', level=None):
    super(Dataset, self).__init__()
    self.split = split
    self.level = level
    self.w, self.h = data_args['w'], data_args['h']
    self.data = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'1data.flist', dtype=np.str_, encoding='utf-8')]
    
    self.gt = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'1gt.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_02 = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'02data.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_02_gt = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'02gt.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_04 = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'04data.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_04_gt = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'04gt.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_06 = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'06data.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_06_gt = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'06gt.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_08 = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'08data.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_08_gt = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'08gt.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_2 = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'2data.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_2_gt = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'2gt.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_4 = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'4data.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_4_gt = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'4gt.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_6 = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'6data.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_6_gt = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'6gt.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_8 = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'8data.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data_8_gt = [os.path.join('../dataset_fine', i)
      for i in np.genfromtxt('../fine_flist/'+split+'8gt.flist', dtype=np.str_, encoding='utf-8')]
    
    self.data.sort()
    self.gt.sort()
    self.data_02.sort()
    self.data_02_gt.sort()
    self.data_04.sort()
    self.data_04_gt.sort()
    self.data_06.sort()
    self.data_06_gt.sort()
    self.data_08.sort()
    self.data_08_gt.sort()
    self.data_2.sort()
    self.data_2_gt.sort()
    self.data_4.sort()
    self.data_4_gt.sort()
    self.data_6.sort()
    self.data_6_gt.sort()
    self.data_8.sort()
    self.data_8_gt.sort()
    
    if split == 'train':
        
       
        temp = np.array([np.hstack((self.data)),np.hstack((self.gt)), np.hstack((self.data_02)), 
                         np.hstack((self.data_02_gt)), np.hstack((self.data_04)), 
                         np.hstack((self.data_04_gt)), np.hstack((self.data_06)), 
                         np.hstack((self.data_06_gt)), np.hstack((self.data_08)), 
                         np.hstack((self.data_08_gt)), np.hstack((self.data_2)), 
                         np.hstack((self.data_2_gt)), np.hstack((self.data_4)), 
                         np.hstack((self.data_4_gt)), np.hstack((self.data_6)), 
                         np.hstack((self.data_6_gt)), np.hstack((self.data_8)), 
                         np.hstack((self.data_8_gt))])
        temp = temp.transpose()
        shuffle(temp)
        self.data = list(temp[:, 0])
        self.gt = list(temp[:, 1])
        self.data_02 = list(temp[:, 2])
        self.data_02_gt = list(temp[:, 3])
        self.data_04 = list(temp[:, 4])
        self.data_04_gt = list(temp[:, 5])
        self.data_06 = list(temp[:, 6])
        self.data_06_gt = list(temp[:, 7])
        self.data_08 = list(temp[:, 8])
        self.data_08_gt = list(temp[:, 9])
        self.data_2 = list(temp[:, 10])
        self.data_2_gt = list(temp[:, 11])
        self.data_4 = list(temp[:, 12])
        self.data_4_gt = list(temp[:, 13])
        self.data_6 = list(temp[:, 14])
        self.data_6_gt = list(temp[:, 15])
        self.data_8 = list(temp[:, 16])
        self.data_8_gt = list(temp[:, 17])

    


    if debug:
      print("Not Support")
      #raise error

      self.data = self.data[:100]
      self.gt = self.gt[:100]

  def __len__(self):
    return len(self.data)
  
  def set_subset(self, start, end):
    self.gt = self.gt[start:end]
    self.data = self.data[start:end]
    self.data_02 = self.data_02[start:end]
    self.data_02_gt = self.data_02_gt[start:end]
    self.data_04 = self.data_04[start:end]
    self.data_04_gt = self.data_04_gt[start:end]
    self.data_06 = self.data_06[start:end]
    self.data_06_gt = self.data_06_gt[start:end]
    self.data_08 = self.data_08[start:end]
    self.data_08_gt = self.data_08_gt[start:end]
    self.data_2 = self.data_2[start:end]
    self.data_2_gt = self.data_2_gt[start:end]
    self.data_4 = self.data_4[start:end]
    self.data_4_gt = self.data_4_gt[start:end]
    self.data_6 = self.data_6[start:end]
    self.data_6_gt = self.data_6_gt[start:end]
    self.data_8 = self.data_8[start:end]
    self.data_8_gt = self.data_8_gt[start:end] 

  def __getitem__(self, index):
    try:
      item = self.load_item(index)
    except:
      print('loading error: ' + self.data[index])
      item = self.load_item(0)
    return item

  def load_item(self, index):
    img_path = os.path.dirname(self.data[index])
    img_name = os.path.basename(self.data[index])
    img = Image.open(os.path.join( img_path, img_name)).convert('RGB')
    # load gt
    gt_path = os.path.dirname(self.gt[index])
    gt_name = os.path.basename(self.gt[index])
    gt = Image.open(os.path.join( gt_path, gt_name)).convert('RGB')

    #load data_02
    data_02_path = os.path.dirname(self.data_02[index])
    data_02_name = os.path.basename(self.data_02[index])
    data_02 = Image.open(os.path.join(data_02_path, data_02_name)).convert('RGB')

    # load data_02_gt
    data_02_gt_path = os.path.dirname(self.data_02_gt[index])
    data_02_gt_name = os.path.basename(self.data_02_gt[index])
    data_02_gt = Image.open(os.path.join(data_02_gt_path, data_02_gt_name)).convert('RGB')

    # load data_04
    data_04_path = os.path.dirname(self.data_04[index])
    data_04_name = os.path.basename(self.data_04[index])
    data_04 = Image.open(os.path.join(data_04_path, data_04_name)).convert('RGB')

    # load data_04_gt
    data_04_gt_path = os.path.dirname(self.data_04_gt[index])
    data_04_gt_name = os.path.basename(self.data_04_gt[index])
    data_04_gt = Image.open(os.path.join(data_04_gt_path, data_04_gt_name)).convert('RGB')

    # load data_06
    data_06_path = os.path.dirname(self.data_06[index])
    data_06_name = os.path.basename(self.data_06[index])
    data_06 = Image.open(os.path.join(data_06_path, data_06_name)).convert('RGB')

    # load data_06_gt
    data_06_gt_path = os.path.dirname(self.data_06_gt[index])
    data_06_gt_name = os.path.basename(self.data_06_gt[index])
    data_06_gt = Image.open(os.path.join(data_06_gt_path, data_06_gt_name)).convert('RGB')

    # load data_08
    data_08_path = os.path.dirname(self.data_08[index])
    data_08_name = os.path.basename(self.data_08[index])
    data_08 = Image.open(os.path.join(data_08_path, data_08_name)).convert('RGB')

    # load data_08_gt
    data_08_gt_path = os.path.dirname(self.data_08_gt[index])
    data_08_gt_name = os.path.basename(self.data_08_gt[index])
    data_08_gt = Image.open(os.path.join(data_08_gt_path, data_08_gt_name)).convert('RGB')

    # load data2
    data_2_path = os.path.dirname(self.data_2[index])
    data_2_name = os.path.basename(self.data_2[index])
    data_2 = Image.open(os.path.join(data_2_path, data_2_name)).convert('RGB')

    # load data2_gt
    data_2_gt_path = os.path.dirname(self.data_2_gt[index])
    data_2_gt_name = os.path.basename(self.data_2_gt[index])
    data_2_gt = Image.open(os.path.join( data_2_gt_path, data_2_gt_name)).convert('RGB')

    # load data4
    data_4_path = os.path.dirname(self.data_4[index])
    data_4_name = os.path.basename(self.data_4[index])
    data_4 = Image.open(os.path.join(data_4_path, data_4_name)).convert('RGB')

    # load data4_gt
    data_4_gt_path = os.path.dirname(self.data_4_gt[index])
    data_4_gt_name = os.path.basename(self.data_4_gt[index])
    data_4_gt = Image.open(os.path.join(data_4_gt_path, data_4_gt_name)).convert('RGB')

    # load data6
    data_6_path = os.path.dirname(self.data_6[index])
    data_6_name = os.path.basename(self.data_6[index])  
    data_6 = Image.open(os.path.join(data_6_path, data_6_name)).convert('RGB')

    # load data6_gt
    data_6_gt_path = os.path.dirname(self.data_6_gt[index])
    data_6_gt_name = os.path.basename(self.data_6_gt[index])
    data_6_gt = Image.open(os.path.join(data_6_gt_path, data_6_gt_name)).convert('RGB')

    # load data8
    data_8_path = os.path.dirname(self.data_8[index])
    data_8_name = os.path.basename(self.data_8[index])
    data_8 = Image.open(os.path.join(data_8_path, data_8_name)).convert('RGB')

    # load data8_gt
    data_8_gt_path = os.path.dirname(self.data_8_gt[index])
    data_8_gt_name = os.path.basename(self.data_8_gt[index])
    data_8_gt = Image.open(os.path.join(data_8_gt_path, data_8_gt_name)).convert('RGB')


    img = img.resize((self.w, self.h))
    gt = gt.resize((self.w, self.h))
    data_02 = data_02.resize((self.w, self.h))
    data_02_gt = data_02_gt.resize((self.w, self.h))
    data_04 = data_04.resize((self.w, self.h))
    data_04_gt = data_04_gt.resize((self.w, self.h))
    data_06 = data_06.resize((self.w, self.h))
    data_06_gt = data_06_gt.resize((self.w, self.h))
    data_08 = data_08.resize((self.w, self.h))
    data_08_gt = data_08_gt.resize((self.w, self.h))
    data_2 = data_2.resize((self.w, self.h))
    data_2_gt = data_2_gt.resize((self.w, self.h))
    data_4 = data_4.resize((self.w, self.h))
    data_4_gt = data_4_gt.resize((self.w, self.h))
    data_6 = data_6.resize((self.w, self.h))
    data_6_gt = data_6_gt.resize((self.w, self.h))
    data_8 = data_8.resize((self.w, self.h))
    data_8_gt = data_8_gt.resize((self.w, self.h))
    #return F.to_tensor(img)*2-1., F.to_tensor(gt)*2-1., img_name, gt_name
    return [[F.to_tensor(img)*2-1., F.to_tensor(gt)*2-1.], [F.to_tensor(data_02)*2-1., F.to_tensor(data_02_gt)*2-1.], [F.to_tensor(data_04)*2-1., F.to_tensor(data_04_gt)*2-1.], [F.to_tensor(data_06)*2-1., F.to_tensor(data_06_gt)*2-1.], [F.to_tensor(data_08)*2-1., F.to_tensor(data_08_gt)*2-1.], [F.to_tensor(data_2)*2-1., F.to_tensor(data_2_gt)*2-1.], [F.to_tensor(data_4)*2-1., F.to_tensor(data_4_gt)*2-1.], [F.to_tensor(data_6)*2-1., F.to_tensor(data_6_gt)*2-1.], [F.to_tensor(data_8)*2-1., F.to_tensor(data_8_gt)*2-1.]]

  def create_iterator(self, batch_size):
    while True:
      sample_loader = DataLoader(dataset=self,batch_size=batch_size,drop_last=True)
      for item in sample_loader:
        yield item
