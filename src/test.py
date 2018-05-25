# speed up the loading of the training data
import cv2
import torch as th
from model import NetG, NetD, NetA
from data_set import CASIABDatasetForTest
import visdom
from torchvision.utils import make_grid

vis = visdom.Visdom(port=5274)
win1 = None
netg = NetG(nc=1)
netd = NetD(nc=1)
neta = NetA(nc=1)
device = th.device("cuda:0")
netg = netg.to(device)
netd = netd.to(device)
neta = neta.to(device)
dataset = CASIABDatasetForTest(
    data_dir='../data/GEI_CASIA_B/gei/')
fineSize = 64

checkpoint = './snapshot_16500.t7'
checkpoint = th.load(checkpoint)
neta.load_state_dict(checkpoint['netA'])
netg.load_state_dict(checkpoint['netG'])
netd.load_state_dict(checkpoint['netD'])
neta.eval()
netg.eval()
netd.eval()

ass_label, noass_label, img = dataset.getbatch(128)
ass_label = ass_label.to(device).to(th.float32)
noass_label = noass_label.to(device).to(th.float32)
img = img.to(device).to(th.float32)
with th.no_grad():
    fake = netg(img)
    fake = (fake + 1) / 2 * 255
    real = (ass_label + 1) / 2 * 255
    ori = (img + 1) / 2 * 255
    al = th.cat((fake, real, ori), 2)
    display = make_grid(al, 20).cpu().numpy()
    if win1 is None:
        win1 = vis.image(display, opts=dict(title="test", caption='test'))
    else:
        vis.image(display, win=win1)
