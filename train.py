#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
import math
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils.config import Config
from torch.utils.data import DataLoader
from utils.dataloader import yolo_dataset_collate, YoloDataset
from nets.yolo_training import YOLOLoss,Generator
from nets.yolo3 import YoloBody
from tqdm import tqdm
import argparse

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            optimizer.zero_grad()
            outputs = net(images)
            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item[0])
            loss = sum(losses)
            loss.backward()
            optimizer.step()

            total_loss += loss
            waste_time = time.time() - start_time
            
            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1), 
                                'lr'        : get_lr(optimizer),
                                'step/s'    : waste_time})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item[0])
                loss = sum(losses)
                val_loss += loss
            pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)
            
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

def init_weights(model):
    # 进行权值初始化
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_epoch", type=int, default=0, help="start number of epoch")
    parser.add_argument("--end_epoch", type=int, default=20, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="number of epochs")
    parser.add_argument("--gamma", type=float, default=0.95, help="number of epochs")
    parser.add_argument("--freeze", type=bool, default=True, help="是否将前面冻结训练")
    parser.add_argument("--annotation_path", type=str, default='tt100k_train.txt', help="train.txt")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--pretrained_weights", type=str, default="logs/yolo_weights.pth",
                        help="if specified starts from checkpoint model")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()

    # 参数初始化,权值初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YoloBody(Config).to(device)
    model = init_weights(model)
    # 加载预训练权重
    if opt.pretrained_weights:
        print('Loading weights into state dict...')
        if opt.pretrained_weights.endswith(".pth"):
            model_dict = model.state_dict()
            pretrained_dict = torch.load(opt.pretrained_weights)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            # model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)
        print('Finished!')

    net = model.train()
    # 使用cuda
    Cuda = torch.cuda.is_available()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"],[-1,2]),
                                    Config["yolo"]["classes"], (Config["img_w"], Config["img_h"]), Cuda))

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(opt.annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # 优化器,最开始使用1e-3的学习率可以收敛的更快
    lr = opt.lr * opt.gamma ** (2 * opt.start_epoch)  # 初始化学习率
    print(lr)
    optimizer = optim.Adam(net.parameters(), lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.gamma)

    train_dataset = YoloDataset(lines[:num_train], (Config["img_h"], Config["img_w"]))
    val_dataset = YoloDataset(lines[num_train:], (Config["img_h"], Config["img_w"]))
    gen = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate)
    gen_val = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)

    epoch_size = num_train // opt.batch_size
    epoch_size_val = num_val // opt.batch_size

    if opt.freeze:
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(opt.start_epoch, opt.end_epoch):
            fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, opt.end_epoch, Cuda)
            lr_scheduler.step()
    else:
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(opt.start_epoch, opt.end_epoch):
            fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, opt.end_epoch, Cuda)
            lr_scheduler.step()