# -*- coding: utf-8 -*-

'''
The Capsules layer.
@author: Yuxian Meng
'''
# TODO: use less permute() and contiguous()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import random
import os

import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchnet.engine import Engine

torch.manual_seed(1991)
torch.cuda.manual_seed(1991)
random.seed(1991)
np.random.seed(1991)


def print_mat(x):
    for i in range(x.size(1)):
        plt.matshow(x[0, i].data.cpu().numpy())

    plt.show()


class PrimaryCaps(nn.Module):
    """
    Primary Capsule layer is nothing more than concatenate several convolutional
    layer together.
    Args:
        A:input channel
        B:number of types of capsules.

    """

    def __init__(self, A=32, B=32):
        super(PrimaryCaps, self).__init__()
        self.B = B
        self.capsules_pose = nn.ModuleList([nn.Conv2d(in_channels=A, out_channels=4 * 4,
                                                      kernel_size=1, stride=1)
                                            for _ in range(self.B)])
        self.capsules_activation = nn.ModuleList([nn.Conv2d(in_channels=A, out_channels=1,
                                                            kernel_size=1, stride=1) for _
                                                  in range(self.B)])

    def forward(self, x):  # b,14,14,32
        poses = [self.capsules_pose[i](x) for i in range(self.B)]  # (b,16,12,12) *32
        poses = torch.cat(poses, dim=1)  # b,16*32,12,12
        activations = [self.capsules_activation[i](x) for i in range(self.B)]  # (b,1,12,12)*32
        activations = F.sigmoid(torch.cat(activations, dim=1))  # b,32,12,12
        return poses, activations


class ConvCaps(nn.Module):
    """
    Convolutional Capsule Layer.
    Args:
        B:input number of types of capsules.
        C:output number of types of capsules.
        kernel: kernel of convolution. kernel=0 means the capsules in layer L+1's
        receptive field contain all capsules in layer L. Kernel=0 is used in the
        final ClassCaps layer.
        stride:stride of convolution
        iteration: number of EM iterations
        coordinate_add: whether to use Coordinate Addition
        transform_share: whether to share transformation matrix.

    """

    def __init__(self, B=32, C=32, kernel=3, stride=2, iteration=3,
                 coordinate_add=False, transform_share=False):
        super(ConvCaps, self).__init__()
        self.B = B
        self.C = C
        self.K = kernel  # kernel = 0 means full receptive field like class capsules
        self.Bkk = None
        self.Cww = None
        self.b = args.batch_size
        self.stride = stride
        self.coordinate_add = coordinate_add
        self.transform_share = transform_share
        self.beta_v = None
        self.beta_a = None
        if not transform_share:
            self.W = nn.Parameter(torch.randn(B, kernel, kernel, C,
                                              4, 4))  # B,K,K,C,4,4
        else:
            self.W = nn.Parameter(torch.randn(B, C, 4, 4))  # B,C,4,4

        self.iteration = iteration

    def coordinate_addition(self, width_in, votes):
        add = [[i / width_in, j / width_in] for i in range(width_in) for j in range(width_in)]  # K,K,w,w
        add = Variable(torch.Tensor(add).cuda()).view(1, 1, self.K, self.K, 1, 1, 1, 2)
        add = add.expand(self.b, self.B, self.K, self.K, self.C, 1, 1, 2).contiguous()
        votes[:, :, :, :, :, :, :, :2, -1] = votes[:, :, :, :, :, :, :, :2, -1] + add
        return votes

    def down_w(self, w):
        return range(w * self.stride, w * self.stride + self.K)

    def EM_routing(self, lambda_, a_, V):
        # routing coefficient
        R = Variable(torch.ones([self.b, self.Bkk, self.Cww]), requires_grad=False).cuda() / self.Cww

        for i in range(self.iteration):
            # M-step
            R = (R * a_)[..., None]
            sum_R = R.sum(1)
            mu = ((R * V).sum(1) / sum_R)[:, None, :, :]
            sigma_square = (R * (V - mu) ** 2).sum(1) / sum_R

            # E-step
            if i != self.iteration - 1:
                mu, sigma_square, V_, a__ = mu.data, sigma_square.data, V.data, a_.data
                normal = Normal(mu, sigma_square[:, None, :, :] ** (1 / 2))
                p = torch.exp(normal.log_prob(V_))
                ap = a__ * p.sum(-1)
                R = Variable(ap / torch.sum(ap, -1)[..., None], requires_grad=False)
            else:
                const = (self.beta_v.expand_as(sigma_square) + torch.log(sigma_square)) * sum_R
                a = torch.sigmoid(lambda_ * (self.beta_a.repeat(self.b, 1) - const.sum(2)))

        return a, mu

    def angle_routing(self, lambda_, a_, V):
        # routing coefficient
        R = Variable(torch.zeros([self.b, self.Bkk, self.Cww]), requires_grad=False).cuda()

        for i in range(self.iteration):
            R = F.softmax(R, dim=1)
            R = (R * a_)[..., None]
            sum_R = R.sum(1)
            mu = ((R * V).sum(1) / sum_R)[:, None, :, :]

            if i != self.iteration - 1:
                u_v = mu.permute(0, 2, 1, 3) @ V.permute(0, 2, 3, 1)
                u_v = u_v.squeeze().permute(0, 2, 1) / V.norm(2, -1) / mu.norm(2, -1)
                R = R.squeeze() + u_v
            else:
                sigma_square = (R * (V - mu) ** 2).sum(1) / sum_R
                const = (self.beta_v.expand_as(sigma_square) + torch.log(sigma_square)) * sum_R
                a = torch.sigmoid(lambda_ * (self.beta_a.repeat(self.b, 1) - const.sum(2)))

        return a, mu

    def forward(self, x, lambda_):
        poses, activations = x
        width_in = poses.size(2)
        w = int((width_in - self.K) / self.stride + 1) if self.K else 1  # 5
        self.Cww = w * w * self.C
        self.b = poses.size(0)

        if self.beta_v is None:
            self.beta_v = nn.Parameter(torch.randn(1, self.Cww, 1)).cuda()
            self.beta_a = nn.Parameter(torch.randn(1, self.Cww)).cuda()

        if self.transform_share:
            if self.K == 0:
                self.K = width_in  # class Capsules' kernel = width_in
            W = self.W.view(self.B, 1, 1, self.C, 4, 4).expand(self.B, self.K, self.K, self.C, 4, 4).contiguous()
        else:
            W = self.W  # B,K,K,C,4,4

        self.Bkk = self.K * self.K * self.B

        # used to store every capsule i's poses in each capsule c's receptive field
        pose = poses.contiguous()  # b,16*32,12,12
        pose = pose.view(self.b, 16, self.B, width_in, width_in).permute(0, 2, 3, 4, 1).contiguous()  # b,B,12,12,16
        poses = torch.stack([pose[:, :, self.stride * i:self.stride * i + self.K,
                             self.stride * j:self.stride * j + self.K, :] for i in range(w) for j in range(w)],
                            dim=-1)  # b,B,K,K,w*w,16
        poses = poses.view(self.b, self.B, self.K, self.K, 1, w, w, 4, 4)  # b,B,K,K,1,w,w,4,4
        W_hat = W[None, :, :, :, :, None, None, :, :]  # 1,B,K,K,C,1,1,4,4
        votes = W_hat @ poses  # b,B,K,K,C,w,w,4,4

        if self.coordinate_add:
            votes = self.coordinate_addition(width_in, votes)
            activation = activations.view(self.b, -1)[..., None].repeat(1, 1, self.Cww)
        else:
            activations_ = [activations[:, :, self.down_w(x), :][:, :, :, self.down_w(y)]
                            for x in range(w) for y in range(w)]
            activation = torch.stack(
                activations_, dim=4).view(self.b, self.Bkk, 1, -1) \
                .repeat(1, 1, self.C, 1).view(self.b, self.Bkk, self.Cww)

        votes = votes.view(self.b, self.Bkk, self.Cww, 16)
        activations, poses = getattr(self, args.routing)(lambda_, activation, votes)
        return poses.view(self.b, self.C, w, w, -1), activations.view(self.b, self.C, w, w)


class CapsNet(nn.Module):
    def __init__(self, A=32, B=32, C=32, D=32, E=10, r=3):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=A,
                               kernel_size=5, stride=2)
        self.primary_caps = PrimaryCaps(A, B)
        self.convcaps1 = ConvCaps(B, C, kernel=3, stride=2, iteration=r,
                                  coordinate_add=False, transform_share=False)
        self.convcaps2 = ConvCaps(C, D, kernel=3, stride=1, iteration=r,
                                  coordinate_add=False, transform_share=False)
        self.classcaps = ConvCaps(D, E, kernel=0, stride=1, iteration=r,
                                  coordinate_add=True, transform_share=True)
        self.decoder = nn.Sequential(
            nn.Linear(16 * args.num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, lambda_, y=None):  # b,1,28,28
        x = F.relu(self.conv1(x))  # b,32,12,12
        x = self.primary_caps(x)  # b,32*(4*4+1),12,12
        x = self.convcaps1(x, lambda_)  # b,32*(4*4+1),5,5
        x = self.convcaps2(x, lambda_)  # b,32*(4*4+1),3,3
        p, a = self.classcaps(x, lambda_)  # b,10*16+10

        p = p.squeeze()

        if y is None:
            _, y = a.max(dim=1)
            y = y.squeeze()

        # convert to one hot
        y = Variable(torch.sparse.torch.eye(args.num_classes)).cuda().index_select(dim=0, index=y)

        reconstructions = self.decoder((p * y[:, :, None]).view(p.size(0), -1))

        return a.squeeze(), reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    @staticmethod
    def spread_loss(x, target, m):  # x:b,10 target:b
        loss = F.multi_margin_loss(x, target, p=2, margin=m)
        return loss

    @staticmethod
    def cross_entropy_loss(x, target, m):
        loss = F.cross_entropy(x, target)
        return loss

    @staticmethod
    def margin_loss(x, labels, m):
        left = F.relu(0.9 - x, inplace=True) ** 2
        right = F.relu(x - 0.1, inplace=True) ** 2

        labels = Variable(torch.sparse.torch.eye(args.num_classes).cuda()).index_select(dim=0, index=labels)

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()
        return margin_loss * 1/x.size(0)

    def forward(self, images, output, labels, m, recon):
        main_loss = getattr(self, args.loss)(output, labels, m)

        if args.use_recon:
            recon_loss = self.reconstruction_loss(recon, images)
            main_loss += 0.0005 * recon_loss

        return main_loss


def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    confusion_meter.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CapsNet')

    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-num_epochs', type=int, default=500)
    parser.add_argument('-lr', type=float, default=2e-2)
    parser.add_argument('-clip', type=float, default=5)
    parser.add_argument('-r', type=int, default=3)
    parser.add_argument('-disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('-print_freq', type=int, default=10)
    parser.add_argument('-pretrained', type=str, default="")
    parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                        help='number of output classes (default: 10)')
    parser.add_argument('-gpu', type=int, default=0, help="which gpu to use")
    parser.add_argument('--env-name', type=str, default='main',
                        metavar='N', help='Environment name for displaying plot')
    parser.add_argument('--loss', type=str, default='margin_loss', metavar='N',
                        help='loss to use: cross_entropy_loss, margin_loss, spread_loss')
    parser.add_argument('--routing', type=str, default='angle_routing', metavar='N',
                        help='routing to use: angle_routing, EM_routing')
    parser.add_argument('--use-recon', type=bool, default=True, metavar='N',
                        help='use reconstruction loss or not')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='num of workers to fetch data')
    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda and torch.cuda.is_available()

    use_cuda = args.use_cuda
    lambda_ = 1e-3  # TODO:find a good schedule to increase lambda and m
    m = 0.2

    A, B, C, D, E, r = 64, 8, 16, 16, args.num_classes, args.r  # a small CapsNet
    # A, B, C, D, E, r = 32, 32, 32, 32, args.num_classes, args.r  # a classic CapsNet

    model = CapsNet(A, B, C, D, E, r)
    capsule_loss = CapsuleLoss()

    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(args.num_classes, normalized=True)

    setting_logger = VisdomLogger('text', opts={'title': 'Settings'}, env=args.env_name)
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'}, env=args.env_name)
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'}, env=args.env_name)
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'}, env=args.env_name)
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'}, env=args.env_name)
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(args.num_classes)),
                                                     'rownames': list(range(args.num_classes))}, env=args.env_name)
    ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'}, env=args.env_name)
    reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'}, env=args.env_name)

    weight_folder = 'weights/{}'.format(args.env_name.replace(' ', '_'))
    if not os.path.isdir(weight_folder):
        os.mkdir(weight_folder)

    setting_logger.log(str(args))

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)

    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              shuffle=True)

    steps, lambda_, m = len(train_dataset) // args.batch_size, 1e-3, 0.2

    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
        m = 0.8
        lambda_ = 0.9

    with torch.cuda.device(args.gpu):
        if use_cuda:
            print("activating cuda")
            model.cuda()

        for epoch in range(args.num_epochs):
            reset_meters()

            # Train
            print("Epoch {}".format(epoch))
            step = 0
            correct = 0
            loss = 0

            with tqdm(total=steps) as pbar:
                for data in train_loader:
                    step += 1
                    if lambda_ < 1:
                        lambda_ += 2e-1 / steps
                    if m < 0.9:
                        m += 2e-1 / steps

                    optimizer.zero_grad()

                    imgs, labels = data  # b,1,28,28; #b
                    imgs, labels = Variable(imgs), Variable(labels)
                    if use_cuda:
                        imgs = imgs.cuda()
                        labels = labels.cuda()

                    out_labels, recon = model(imgs, lambda_, labels)

                    recon = recon.view_as(imgs)
                    loss = capsule_loss(imgs, out_labels, labels, m, recon)

                    loss.backward()
                    optimizer.step()

                    meter_accuracy.add(out_labels.data, labels.data)
                    meter_loss.add(loss.data[0])
                    pbar.set_postfix(loss=meter_loss.value()[0], acc=meter_accuracy.value()[0])
                    pbar.update()

                loss = meter_loss.value()[0]
                acc = meter_accuracy.value()[0]

                train_loss_logger.log(epoch, loss)
                train_error_logger.log(epoch, acc)

                print("Epoch{} Train acc:{:4}, loss:{:4}".format(epoch, acc, loss))
                scheduler.step(acc)
                torch.save(model.state_dict(), "./weights/em_capsules/model_{}.pth".format(epoch))

                reset_meters()
                # Test
                print('Testing...')
                correct = 0
                for i, data in enumerate(test_loader):
                    imgs, labels = data  # b,1,28,28; #b
                    imgs, labels = Variable(imgs, volatile=True), Variable(labels, volatile=True)
                    if use_cuda:
                        imgs = imgs.cuda()
                        labels = labels.cuda()
                    out_labels, recon = model(imgs, lambda_)  # b,10,17

                    recon = imgs.view_as(imgs)
                    loss = capsule_loss(imgs, out_labels, labels, m, recon)

                    # visualize reconstruction for final batch
                    if i == 0:
                        ground_truth_logger.log(
                            make_grid(imgs.data, nrow=int(args.batch_size ** 0.5), normalize=True,
                                      range=(0, 1)).cpu().numpy())
                        reconstruction_logger.log(
                            make_grid(recon.data, nrow=int(args.batch_size ** 0.5), normalize=True,
                                      range=(0, 1)).cpu().numpy())

                    meter_accuracy.add(out_labels.data, labels.data)
                    confusion_meter.add(out_labels.data, labels.data)
                    meter_loss.add(loss.data[0])

                loss = meter_loss.value()[0]
                acc = meter_accuracy.value()[0]

                test_loss_logger.log(epoch, loss)
                test_accuracy_logger.log(epoch, acc)
                confusion_logger.log(confusion_meter.value())

                print("Epoch{} Test acc:{:4}, loss:{:4}".format(epoch, acc, loss))

