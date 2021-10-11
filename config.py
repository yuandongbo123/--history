# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser(description='PET/MRI Challenge')

# data in/out and dataset
parser.add_argument('--name', type=float, default=0.5,
                    help='transforms.ColorJitter')
# parser.add_argument('--dataset_train_path', default='D:/YUANDONGBO/pet_mri/enhancement/train/*/*/*.png',
#                     help='train root path')
parser.add_argument('--dataset_test_path', default='/home/tione/notebook/dcm2png_test',
                    help='test root path')
parser.add_argument("--save_dir", default='/home/tione/notebook/train/v0920_b8', help="all data dir")  # 运行产生的所有文件保存在此文件夹下
parser.add_argument('--v', type=str, default='v7',
                    help='version')
# 图像增强
parser.add_argument('--ColorJitter', type=float, default=0.5,
                    help='transforms.ColorJitter')
parser.add_argument('--RandomRotation', type=int, default=180,
                    help='transforms.RandomRotation')
parser.add_argument('--Resize', type=int, default=512,
                    help='transforms.Resize')
# train
parser.add_argument('--batch_size', type=list, default=10,
                    help='batch size of trainset')
parser.add_argument('--k', type=int, default=5,
                    help='Cross-validation')
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='gamma')
parser.add_argument('--step_size', type=int, default=10,
                    help='step_size')
# resume 中断训练后恢复训练
parser.add_argument('--resume', type=bool, default=False,
                    help='resume')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start_epoch')
parser.add_argument('--loss', default='MSELoss',
                        choices=['CrossEntropyLoss', 'FocalLoss', 'MSELoss', 'multitask'])
parser.add_argument('--reg_coef', default=0.5, type=float)
parser.add_argument('--cls_coef', default=0.5, type=float)
parser.add_argument('--pred_type', default='regression',
                        choices=['classification', 'regression', 'multitask'])
parser.add_argument('--pseudo_labels', default=None,
                    help='pseudo_labels')
# parser.add_argument('--name', default='efficientb5',
#                      help='model name: (default: arch+timestamp)')
args = parser.parse_args()
