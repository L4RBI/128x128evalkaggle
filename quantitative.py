import metrique
import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from  piqa import MS_SSIM
from costumDataset import Kaiset,Kaiset2
import sys
from torchvision.utils import save_image
#chooses what model to train
if config.MODEL == "ResUnet":
    from resUnet import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from time import localtime
import os
if not os.path.exists("evaluation"):
    os.mkdir("evaluation")
writer=SummaryWriter("train{}-{}".format(localtime().tm_mon,localtime().tm_mday))
torch.backends.cudnn.benchmark = True
if not os.path.exists(sys.argv[9]):
    os.mkdir(sys.argv[9])
if not os.path.exists(sys.argv[9]+"/image"):
    os.mkdir(sys.argv[9]+"/image")
if not os.path.exists("original"):
    os.mkdir("original")

def test_fn(
     gen, loader, metric, epoch=0
):
    loop = tqdm(loader, leave=True)
    gen.eval()
    with torch.no_grad():
     resultat=[]
     resultat2 = []
     resultat3 = []
     for idx, (x, y,z) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator


        y_fake = gen(x)

        y_fake=y_fake*0.5+0.5
        L1 = metric[0](y_fake, y)
        ssim = metric[1](y_fake.float(), y.float())
        mse =metric[2](y_fake*255, y*255)
        resultat.append(L1.item()*255)
        resultat2.append(ssim.item())
        resultat3.append(mse.item())

        if idx % 1 == 0:
            loop.set_postfix(

                L1    =L1.item()*255,
                #ssim=ssim.item(),
                mse=mse.item()*255
            )

    return torch.tensor(resultat).mean(),torch.tensor(resultat2).mean(),torch.tensor(resultat3).mean()
def main():
    #instancing the models
    gen = Generator(init_weight=config.INIT_WEIGHTS).to(config.DEVICE)
    #print(gen)
    #instancing the optims
    #instancing the Loss-functions

    L1_LOSS = nn.L1Loss()

    ssim = MS_SSIM(window_size=8, n_channels=3, value_range=1.)
    mse=nn.MSELoss()
    #if true loads the checkpoit in the ./
    if sys.argv[6]!="none":
        load_checkpoint(
            sys.argv[6], gen, config.LEARNING_RATE,
        )

    #training data loading
    test_dataset = Kaiset2(path=sys.argv[1],train=False, Listset=config.DTRAIN_LIST if sys.argv[5]=="0"else config.NTRAIN_LIST)
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(sys.argv[4]),
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    #enabling MultiPrecision Mode, the optimise performance
    #evauation data loading

    for epoch in range(1):
        x=test_fn(gen, test_loader,  [L1_LOSS,ssim,mse],epoch=epoch)
        file=open(sys.argv[9]+"/resultat.txt",'w')
        file.write("L1:"+str(x[0])+"\n")
        file.write("SSIM:"+str(x[1]) + "\n")
        file.write("MSE:"+str(x[2]) + "\n")
if __name__ == "__main__":
    main()