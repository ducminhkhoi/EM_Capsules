# -*- coding: utf-8 -*-

'''
The Capsules layer.
@author: Yuxian Meng
'''
# TODO: use less permute() and contiguous()


import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor, pi
from torch.autograd import Variable
from torch.distributions import Normal
import numpy as np
import random

# from time import time
from torch.optim import lr_scheduler
from torchvision import datasets
import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
torch.manual_seed(1991)
torch.cuda.manual_seed(1991)
random.seed(1991)
np.random.seed(1991)


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
        self.beta_v = nn.Parameter(torch.randn(1))
        self.beta_a = nn.Parameter(torch.randn(1))  # TODO: make sure whether beta_a depend on c
        if not transform_share:
            self.W = nn.Parameter(torch.randn(B, kernel, kernel, C,
                                              4, 4))  # B,K,K,C,4,4
        else:
            self.W = nn.Parameter(torch.randn(B, C, 4, 4))  # B,C,4,4

        self.iteration = iteration

    def coordinate_addition(self, width_in, votes):
        add = [[i/width_in, j/width_in] for i in range(width_in) for j in range(width_in)]  # K,K,w,w
        add = Variable(torch.Tensor(add).cuda()).view(1, 1, self.K, self.K, 1, 1, 1, 2)
        add = add.expand(self.b, self.B, self.K, self.K, self.C, 1, 1, 2).contiguous()
        votes[:, :, :, :, :, :, :, :2, -1] = votes[:, :, :, :, :, :, :, :2, -1] + add
        return votes

    def down_w(self, w):
        return range(w * self.stride, w * self.stride + self.K)

    def EM_routing(self, lambda_, a_, V):
        # routing coefficient
        R = Variable(torch.ones([a_.size(0), self.Bkk, self.Cww]), requires_grad=False).cuda() / self.Cww

        for i in range(self.iteration):
            # M-step
            R = (R * a_)[..., None]
            sum_R = R.sum(1)
            mu = ((R * V).sum(1) / sum_R)[:, None, :, :]
            sigma_square = (R * (V - mu) ** 2).sum(1) / sum_R
            const = (self.beta_v + torch.log(sigma_square)) * sum_R
            a = torch.sigmoid(lambda_*(self.beta_a - const.sum(2)))

            # E-step
            if i != self.iteration - 1:
                mu, sigma_square, V_, a__ = mu.data, sigma_square.data, V.data, a_.data
                normal = Normal(mu, sigma_square[:, None, :, :] ** (1/2))
                p = torch.exp(normal.log_prob(V_))
                ap = a__ * p.sum(-1)
                R = Variable(ap / torch.sum(ap, -1)[..., None], requires_grad=False)

        return a, mu

    def forward(self, x, lambda_):
        poses, activations = x
        width_in = poses.size(2)
        w = int((width_in - self.K) / self.stride + 1) if self.K else 1  # 5

        if self.transform_share:
            if self.K == 0:
                self.K = width_in  # class Capsules' kernel = width_in
            W = self.W.view(self.B, 1, 1, self.C, 4, 4).expand(self.B, self.K, self.K, self.C, 4, 4).contiguous()
        else:
            W = self.W  # B,K,K,C,4,4

        self.Cww = w * w * self.C
        self.Bkk = self.K * self.K * self.B
        self.b = poses.size(0)

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
            activation = torch.stack(activations_, dim=4).view(self.b, self.Bkk, 1, -1) \
                .repeat(1, 1, self.C, 1).view(self.b, self.Bkk, self.Cww)

        votes = votes.view(self.b, self.Bkk, self.Cww, 16)
        activations, poses = self.EM_routing(lambda_, activation, votes)
        return poses.view(self.b, self.C, w, w, -1), activations.view(self.b, self.C, w, w)


def print_mat(x):
    for i in range(x.size(1)):
        plt.matshow(x[0, i].data.cpu().numpy())

    plt.show()


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
        self.num_class = E

    def forward(self, x, lambda_):  # b,1,28,28
        x_c = F.relu(self.conv1(x))  # b,32,12,12
        x_p = self.primary_caps(x_c)  # b,32*(4*4+1),12,12
        x_cc1 = self.convcaps1(x_p, lambda_)  # b,32*(4*4+1),5,5
        x_cc2 = self.convcaps2(x_cc1, lambda_)  # b,32*(4*4+1),3,3
        x_cc = self.classcaps(x_cc2, lambda_)  # b,10*16+10
        return x_cc[0], x_cc[1].view(-1, 10)

    def loss(self, x, target, m):  # x:b,10 target:b
        one_shot_target = Variable(torch.sparse.torch.eye(self.num_class)).cuda().index_select(dim=0, index=target)
        a_t = torch.sum(x * one_shot_target, dim=1, keepdim=True)
        loss = torch.sum(F.relu(m - (a_t - x)) ** 2, dim=1) - m ** 2
        return 1/x.size(0)*loss.sum()

    def loss2(self, x, target):
        loss = F.cross_entropy(x, target)
        return loss


def get_dataloader(args):
    # MNIST Dataset
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
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True)

    return train_loader, test_loader


def get_args():
    parser = argparse.ArgumentParser(description='CapsNet')

    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-num_epochs', type=int, default=500)
    parser.add_argument('-lr', type=float, default=2e-2)
    parser.add_argument('-clip', type=float, default=5)
    parser.add_argument('-r', type=int, default=3)
    parser.add_argument('-disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('-print_freq', type=int, default=10)
    parser.add_argument('-pretrained', type=str, default="")
    parser.add_argument('-gpu', type=int, default=1, help="which gpu to use")
    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda and torch.cuda.is_available()

    return args


if __name__ == '__main__':
    args = get_args()
    train_loader, test_loader = get_dataloader(args)
    use_cuda = args.use_cuda
    steps = len(train_loader.dataset) // args.batch_size
    lambda_ = 1e-3  # TODO:find a good schedule to increase lambda and m
    m = 0.2
    A, B, C, D, E, r = 64, 8, 16, 16, 10, args.r  # a small CapsNet
    #    A,B,C,D,E,r = 32,32,32,32,10,args.r # a classic CapsNet

    is_nan = False
    model = CapsNet(A, B, C, D, E, r)

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    with torch.cuda.device(args.gpu):
        #        print(args.gpu, type(args.gpu))
        if args.pretrained:
            model.load_state_dict(torch.load(args.pretrained))
            m = 0.8
            lambda_ = 0.9
        if use_cuda:
            print("activating cuda")
            model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)
        for epoch in range(args.num_epochs):
            # Train
            print("Epoch {}".format(epoch))
            step = 0
            correct = 0
            loss = 0
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

                _, out_labels = model(imgs, lambda_)

                loss = model.loss(out_labels, labels, m)
                # loss = model.loss2(out_labels, labels)

                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
                loss.backward()
                optimizer.step()

                # stats
                pred = out_labels.max(1)[1]  # b
                acc = pred.eq(labels).cpu().sum().data[0]
                correct += acc
                if step % args.print_freq == 0:
                    print("batch:{}, loss:{:.4f}, acc:{:}/{}".format(
                        step, loss.data[0], acc, args.batch_size))
            acc = correct / len(train_loader.dataset)
            print("Epoch{} Train acc:{:4}".format(epoch, acc))
            scheduler.step(acc)
            torch.save(model.state_dict(), "./weights/em_capsules/model_{}.pth".format(epoch))

            # Test
            print('Testing...')
            correct = 0
            for data in test_loader:
                imgs, labels = data  # b,1,28,28; #b
                imgs, labels = Variable(imgs, volatile=True), Variable(labels, volatile=True)
                if use_cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                _, out_labels = model(imgs, lambda_)  # b,10,17
                # loss = model.loss(out_labels, labels, m)
                # stats
                pred = out_labels.max(1)[1]  # b
                acc = pred.eq(labels).cpu().sum().data[0]
                correct += acc
            acc = correct / len(test_loader.dataset)
            print("Epoch{} Test acc:{:4}".format(epoch, acc))
