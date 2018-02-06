This is a Pytorch implementation of EM Capsules in the paper "Matrix capsules with EM routing"

The code is based on this repository: https://github.com/shzygmyx/Matrix-Capsules-pytorch.

You need to install pytorch.tnt for logger and visualization, follow instructions on [`https://github.com/pytorch/tnt`](https://github.com/pytorch/tnt)
and Visdom, follow instructions on [`https://github.com/facebookresearch/visdom`](https://github.com/facebookresearch/visdom)

Some improvements:
+ Big improvement is that replacing all for loops in routing by matrix multiplication
+ Use visdom to log and visualize learning and testing phases
+ Add more losses: cross_entropy_loss, margin_loss (from Dynamic Routing Between Capsules paper), reconstruction_loss
+ Add more routing: angle_routing (from Dynamic Routing Between Capsules paper)
+ Can use multiple workers at the same time to load data much faster

Therefore, the performance is much better than the original code.

For instruction, read the main.py for options in argparse and A, B, C, D when training.