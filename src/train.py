# speed up the loading of the training data
import cv2
import numpy as np
import torch as th
import itertools
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model import NetG, NetD, NetA
from data_set import CASIABDataset
import torch.optim as optim
import visdom
from torchvision.utils import make_grid

vis = visdom.Visdom(port=5274)
win = None
win1 = None
netg = NetG(nc=1)
netd = NetD(nc=1)
neta = NetA(nc=1)
device = th.device("cuda:2")

# weights init
all_mods = itertools.chain()
all_mods = itertools.chain(all_mods, [
    list(netg.children())[0].children(),
    list(netd.children())[0].children(),
    list(neta.children())[0].children()
])
for mod in all_mods:
    if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.ConvTranspose2d):
        init.normal_(mod.weight, 0.0, 0.02)
    elif isinstance(mod, nn.BatchNorm2d):
        init.normal_(mod.weight, 1.0, 0.02)
        init.constant_(mod.bias, 0.0)

netg = netg.to(device)
netd = netd.to(device)
neta = neta.to(device)
netg.train()
netd.train()
neta.train()
dataset = CASIABDataset(data_dir='../data/GEI_CASIA_B/gei/')

iteration = 0
lr = 0.0002
real_label = 1
fake_label = 0
fineSize = 64

label = th.zeros((128, 1), requires_grad=False).to(device)
optimG = optim.Adam(netg.parameters(), lr=lr/2)
optimD = optim.Adam(netd.parameters(), lr=lr/3)
optimA = optim.Adam(neta.parameters(), lr=lr/3)
print('Training starts')
while iteration < 1000000:
    ass_label, noass_label, img = dataset.getbatch(128)
    ass_label = ass_label.to(device).to(th.float32)
    noass_label = noass_label.to(device).to(th.float32)
    img = img.to(device).to(th.float32)
    # update D
    lossD = 0
    optimD.zero_grad()
    output = netd(ass_label)
    label.fill_(real_label)
    lossD_real1 = F.binary_cross_entropy(output, label)
    lossD += lossD_real1.item()
    lossD_real1.backward()

    label.fill_(real_label)
    output1 = netd(noass_label)
    lossD_real2 = F.binary_cross_entropy(output1, label)
    lossD == lossD_real2.item()
    lossD_real2.backward()

    fake = netg(img).detach()
    label.fill_(fake_label)
    output2 = netd(fake)

    lossD_fake = F.binary_cross_entropy(output2, label)
    lossD += lossD_fake.item()
    lossD_fake.backward()

    optimD.step()
    # update A
    lossA = 0
    optimA.zero_grad()
    assd = th.cat((img, ass_label), 1)
    noassd = th.cat((img, noass_label), 1)
    fake = netg(img).detach()
    faked = th.cat((img, fake), 1)

    label.fill_(real_label)
    output1 = neta(assd)
    lossA_real1 = F.binary_cross_entropy(output1, label)
    lossA += lossA_real1.item()
    lossA_real1.backward()

    label.fill_(fake_label)
    output = neta(noassd)
    lossA_real2 = F.binary_cross_entropy(output, label)
    lossA += lossA_real2.item()
    lossA_real2.backward()

    label.fill_(fake_label)
    output = neta(faked)
    lossA_fake = F.binary_cross_entropy(output, label)
    lossA += lossA_fake.item()
    lossA_fake.backward()
    optimA.step()
    # update G
    lossG = 0
    optimG.zero_grad()
    fake = netg(img)
    output = netd(fake)

    label.fill_(real_label)
    lossGD = F.binary_cross_entropy(output, label)
    lossG += lossGD.item()
    lossGD.backward(retain_graph=True)

    faked = th.cat((img, fake), 1)
    output = neta(faked)
    label.fill_(real_label)
    lossGA = F.binary_cross_entropy(output, label)
    lossG += lossGA.item()
    lossGA.backward()
    optimG.step()

    iteration += 1

    if iteration % 20 == 0:
        with th.no_grad():
            netg.eval()
            fake = netg(img)
            netg.train()
        fake = (fake + 1) / 2 * 255
        real = (ass_label + 1) / 2 * 255
        ori = (img + 1) / 2 * 255
        al = th.cat((fake, real, ori), 2)
        display = make_grid(al, 20).cpu().numpy()
        if win1 is None:
            win1 = vis.image(display, opts=dict(title="train", caption='train'))
        else:
            vis.image(display, win=win1)
    if iteration % 500 == 0:
        state = {
            'netA': neta.state_dict(),
            'netG': netg.state_dict(),
            'netD': netd.state_dict()
        }
        th.save(state, './snapshot_%d.t7' % iteration)
        print('iter = {}, ErrG = {}, ErrA = {}, ErrD = {}'.format(
            iteration, lossG/2, lossA/3, lossD/3
        ))
    if iteration % 20 == 0:
        if win is None:
            win = vis.line(X=np.array([[iteration, iteration,
                                        iteration]]),
                           Y=np.array([[lossG/2, lossA/3, lossD/3]]),
                           opts=dict(
                               title='GaitGAN',
                               ylabel='loss',
                               xlabel='iterations',
                               legend=['lossG', 'lossA', 'lossD']
                           ))
        else:
            vis.line(X=np.array([[iteration, iteration,
                                  iteration]]),
                     Y=np.array([[lossG/2, lossA/3, lossD/3]]),
                     win=win,
                     update='append')
