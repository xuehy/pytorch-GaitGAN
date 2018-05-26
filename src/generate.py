# speed up the loading of the training data
import cv2
import torch as th
from model import NetG, NetD, NetA
from data_set import CASIABDatasetGenerate
import visdom


vis = visdom.Visdom(port=5274)
win1 = None
netg = NetG(nc=1)
netd = NetD(nc=1)
neta = NetA(nc=1)
device = th.device("cuda:0")
netg = netg.to(device)
netd = netd.to(device)
neta = neta.to(device)
fineSize = 64

checkpoint = './snapshot_28500.t7'
checkpoint = th.load(checkpoint)
neta.load_state_dict(checkpoint['netA'])
netg.load_state_dict(checkpoint['netG'])
netd.load_state_dict(checkpoint['netD'])
neta.eval()
netg.eval()
netd.eval()
angles = ['000', '018', '036', '054', '072', '090',
          '108', '126', '144', '162', '180']
for cond in ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05',
             'nm-06']:
    dataset = CASIABDatasetGenerate(data_dir='../data/GEI_CASIA_B/gei/',
                                    cond=cond)
    for i in range(1, 125):
        ass_label, img = dataset.getbatch(i, 10)
        img = img.to(device).to(th.float32)

        with th.no_grad():
            fake = netg(img)
            fake = (fake + 1) / 2 * 255
            for j in range(10):
                fake_ = fake[j].squeeze().cpu().numpy()
                ang = angles[j]
                cv2.imwrite('./transformed_28500/%03d-%s-%s.png' % (i, cond, ang), fake_)
        
