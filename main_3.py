# -*- coding: utf-8 -*-
import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import time, datetime
import pdb, traceback

import cv2
# import imagehash
from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b4') 

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import config
import math
import pydicom 
from pydicom.pixel_data_handlers.util import apply_voi_lut
import time, datetime
import pdb, traceback
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import metrics
import cv2
from tqdm import tqdm
# import imagehash
from PIL import Image
import random
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b4') 

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 指定第2块gpu
# torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True  # 加速神经网络计算

seed = 21
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed) #cpu随机种子
torch.cuda.manual_seed(seed) #为gpu设置随机数种子
torch.backends.cudnn.deterministic=True

# input dataset
    
# class QRDataset(Dataset):
#     def __init__(self, train_jpg, transform=None):
#         self.train_jpg = train_jpg
#         if transform is not None:
#             self.transform = transform
#         else:
#             self.transform = None
    
#     def __getitem__(self, index):
#         start_time = time.time()
#         img = Image.open(self.train_jpg[index]).convert('RGB')
        
#         if self.transform is not None:
#             img = self.transform(img)
        
#         return img,torch.from_numpy(np.array(int('AD' in self.train_jpg[index])))
    
#     def __len__(self):
#         return len(self.train_jpg)

class QRDataset(Dataset):
    """
        construct the dataset
    """
    def __init__(self,images_path,images_label,transform=None,train=True):
        self.imgs = [os.path.join(img_path,"".join(path +'.png')) for path in images_path]
        # if train dataset : get the appropriate label
        if train:
            self.train = True
            self.labels = images_label
        else:
            self.train = False

        self.transform = transform 
    def load_dicom_image(self,
        img,
        img_size=512,
        scale=0.8):
        '''
        This function allows you to load a DCIM type image 
        and apply preprocessing steps such as crop, resize 
        and denoising filter to it.
        ****************************************************
        PARAMETERS
        ****************************************************
        - path : String
            Path to the DCIM image file to load.
        - img_size : Integer
            Image size desired for resizing.
        - scale : Float
            Desired scale for the cropped image
        - prep : Bool
            True for a full preprocessing with
            denoising.
        '''
        # Load single image
    #     img = pydicom.read_file(path).pixel_array
        # Crop image
        center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
        width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
        left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
        top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
        img = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
        # Resize image
    #     img = cv2.resize(img, (img_size, img_size))

        return img
    def crop_image(self,img, tol=7):
        if img.ndim ==2:
            mask = img>tol
            return img[np.ix_(mask.any(1),mask.any(0))]
        elif img.ndim==3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img>tol
            check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
            if (check_shape == 0): # image is too dark so that we crop out everything,
                return img # return original image
            else:
                img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
                img = np.stack([img1,img2,img3],axis=-1)

            return img
        
    def circle_crop(self,img):
        img = self.crop_image(img)

        height, width, depth = img.shape
        largest_side = np.max((height, width))
        img = cv2.resize(img, (largest_side, largest_side))

        height, width, depth = img.shape

        x = width//2
        y = height//2
        r = np.amin((x, y))

        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)
        img = self.crop_image(img)

        return img    
    def __getitem__(self,index):
#         image_path = self.imgs[index]  +'.png'
        img = cv2.imread(self.imgs[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.circle_crop(img)
#         img = self.load_dicom_image(img)
        # RGB为彩色图片，L为灰度图片
        img = Image.fromarray(img)
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        if self.transform:
            transform = self.transform
        else:
            # if not define the transform:default resize the figure(224,224) and ToTensor
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
                ])
        img_t = transform(img)
        if self.train:
            image_label = self.labels[index]
            return img_t,image_label
        else:
            return img_t 
    def __len__(self):
        return len(self.imgs)

        
        
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""


    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    
    
    
    
class VisitNet(nn.Module):
    def __init__(self):
        super(VisitNet, self).__init__()
                
        model = models.resnet34(True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 5)
        self.resnet = model
        
    def forward(self, img):        
        out = self.resnet(img)
        return out

def CrossEntropyLoss_label_smooth(outputs, targets,
                                  num_classes=5, epsilon=0.1):
    N = targets.size(0)
    smoothed_labels = torch.full(size=(N, num_classes),
                                 fill_value=epsilon / (num_classes - 1))
    targets = targets.data.cpu()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1),
                             value=1 - epsilon)
    # outputs = outputs.data.cpu()
    log_prob = nn.functional.log_softmax(outputs, dim=1).cpu()
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss      
    
    
def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        return top1

def predict(test_loader, model, tta=10):
    # switch to evaluate mode
    model.eval()
    
    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(test_loader):
                input = input.cuda()
                target = target.cuda()

                # compute output
                output = model(input, path)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)
    
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
    
    return test_pred_tta

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
#         loss = criterion(output, target)
        loss = CrossEntropyLoss_label_smooth(output, target, num_classes=5)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        # top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.pr2int(i)
            
            
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """
 
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)
 
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
 
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
 
    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)
 
    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
            
if __name__ == '__main__':
    print(1)
    args = config.args
    df = pd.read_csv('/home/tione/notebook/taop-2021/100001/To user/train1_data_info.csv')
#     img_path = '/home/tione/notebook/taop-2021/100001/To user'
    img_path = '/home/tione/notebook/dcm2png'
#     np.random.seed(2020)
    sampler = np.random.permutation(3000)
    df = df.take(sampler)
    labelencoder = LabelEncoder()
    labelencoder.fit(df['class(1:PM;2:MD;3:glaucoma;4:RVO;5:DR)'])
    df['label'] = labelencoder.transform(df['class(1:PM;2:MD;3:glaucoma;4:RVO;5:DR)'])
    skf = KFold(n_splits=10, random_state=233, shuffle=True)
    for flod_idx, (train_idx, val_idx) in enumerate(skf.split(df['id_patient'])):    
        train_loader = torch.utils.data.DataLoader(
            QRDataset(df['id_patient'][train_idx],
                      np.array(df['label'][train_idx].astype('int64')),
                    transforms.Compose([
                                # transforms.RandomGrayscale(),
                                transforms.RandomRotation(degrees=180, expand=False),
                                transforms.Resize((512, 512)),
                                transforms.RandomAffine(10),
                                # transforms.ColorJitter(hue=.05, saturation=.05),
                                # transforms.RandomCrop((450, 450)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            ), batch_size=10, shuffle=True, num_workers=20, pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            QRDataset(df['id_patient'][val_idx],
                      np.array(df['label'][val_idx].astype('int64')),
                    transforms.Compose([
#                                 transforms.RandomRotation(degrees=60, expand=False),
                                transforms.Resize((512, 512)),
                                # transforms.Resize((124, 124)),
                                # transforms.RandomCrop((450, 450)),
                                # transforms.RandomCrop((88, 88)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            ), batch_size=10, shuffle=False, num_workers=10, pin_memory=True
        )

        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), 'gpus')
            model = VisitNet().cuda()
            model = nn.DataParallel(model,device_ids=[0,1]) ###如果只有一个gpu把1去掉
#         model = VisitNet().cuda()
        # model = nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), 0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(
#                 optimizer, T_max=30, eta_min=1e-5)
#         optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#         scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
#         scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=4, after_scheduler=scheduler_steplr)

        best_acc = 0.0
        for epoch in range(20):
            scheduler.step()
            print('Epoch: ', epoch)

            train(train_loader, model, criterion, optimizer, epoch)
            val_acc = validate(val_loader, model, criterion)

            if val_acc.avg.item() > best_acc:
                best_acc = val_acc.avg.item()
                torch.save(model.state_dict(), './resnet18_fold{0}.pt'.format(flod_idx))

        break
