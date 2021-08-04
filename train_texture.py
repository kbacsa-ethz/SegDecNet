from comet_ml import Experiment

from models import SegmentNet, weights_init_normal
from dataset import TokaidoTextureDataset

import torch

from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import numpy as np
import pandas as pd
import sys
import argparse
import time
import PIL.Image as Image


def main(cfg):
    hyper_params = vars(cfg)
    experiment = Experiment(project_name="learn-texture", api_key="Bm8mJ7xbMDa77te70th8PNcT8", disabled=not cfg.comet)
    experiment.log_parameters(hyper_params)

    seed = 42
    torch.manual_seed(seed)

    # use gpu
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    # Build nets
    segment_net = SegmentNet(init_weights=True)
    segment_net.to(device)

    # Loss functions
    criterion_segment = torch.nn.MSELoss()

    if opt.cuda:
        segment_net = segment_net.cuda()
        criterion_segment.cuda()

    if opt.gpu_num > 1:
        segment_net = torch.nn.DataParallel(segment_net, device_ids=list(range(opt.gpu_num)))

    if opt.begin_epoch != 0:
        # Load pretrained models
        segment_net.load_state_dict(torch.load("./saved_models/segment_net_%d.pth" % (opt.begin_epoch)))
    else:
        # Initialize weights
        segment_net.apply(weights_init_normal)

    # Optimizers
    optimizer_seg = torch.optim.Adam(segment_net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    transforms_ = transforms.Compose([
        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transforms_mask = transforms.Compose([
        transforms.Resize((opt.img_height // 8, opt.img_width // 8)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    df_puretex_train = pd.read_csv(os.path.join(cfg.root_dir, 'files_puretex_train.csv'),
                                   names=['image_path', 'damage_path'])
    df_puretex_train = df_puretex_train.replace(to_replace=r'\\', value='/', regex=True)
    df_puretex_test = pd.read_csv(os.path.join(cfg.root_dir, 'files_puretex_test.csv'),
                                   names=['image_path', 'damage_path'])
    df_puretex_test = df_puretex_test.replace(to_replace=r'\\', value='/', regex=True)

    train_data = TokaidoTextureDataset(
        cfg.root_dir,
        transforms_=transforms_,
        transforms_mask=transforms_mask,
        dataFrame=df_puretex_train
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.worker_num
    )

    test_data = TokaidoTextureDataset(
        cfg.root_dir,
        transforms_=transforms_,
        transforms_mask=transforms_mask,
        dataFrame=df_puretex_test,
        isTrain=False
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.worker_num
    )

    global_step = 0
    for epoch in range(cfg.begin_epoch, cfg.end_epoch):
        segment_net.train()
        batch_num = 0

        for img, mask in train_loader:
            img = img.to(device)
            mask = mask.to(device)

            optimizer_seg.zero_grad()

            rst = segment_net(img)

            seg = rst["seg"]

            loss_seg = criterion_segment(seg, mask)
            loss_seg.backward()
            optimizer_seg.step()
            batch_num += 1
            global_step += 1

            experiment.log_metric("training_loss", loss_seg, step=global_step)

            sys.stdout.write(
                "\r [Epoch %d/%d]  [Batch %d/%d] [loss %f]"
                % (
                    epoch,
                    cfg.end_epoch,
                    batch_num,
                    len(train_data),
                    loss_seg.item()
                )
            )

        # test ****************************************************************************
        if cfg.need_test and epoch % cfg.test_interval == 0 and epoch >= cfg.test_interval:
            segment_net.eval()

            for i, test_img in enumerate(test_loader):
                test_img = test_img.to(device)
                t1 = time.time()
                rst_test = segment_net(test_img)
                t2 = time.time()
                seg_test = rst_test["seg"]

                save_path_str = "./testResultSeg/epoch_%d" % epoch
                if os.path.exists(save_path_str) == False:
                    os.makedirs(save_path_str, exist_ok=True)
                    # os.mkdir(save_path_str)

                print("processing image NO %d, time comsuption %fs" % (i, t2 - t1))
                foo = test_img.squeeze().permute(1, 2, 0).numpy()

                experiment.log_image(image_data=test_img.detach().squeeze().permute(1, 2, 0), name="%s_img_%d.jpg" % (save_path_str, i))
                experiment.log_image(image_data=seg_test.detach().squeeze(), name="%s_seg_%d.jpg" % (save_path_str, i))
                # save_image(imgTest.data, "%s/img_%d.jpg" % (save_path_str, i))
                # save_image(segTest.data, "%s/img_%d_seg.jpg" % (save_path_str, i))

            segment_net.train()

        # save parameters *****************************************************************
        if cfg.need_save and epoch % cfg.save_interval == 0 and epoch >= cfg.save_interval:
            segment_net.eval()

            save_path_str = "./saved_models"
            if os.path.exists(save_path_str) == False:
                os.makedirs(save_path_str, exist_ok=True)

            torch.save(segment_net.state_dict(), "%s/segment_net_%d.pth" % (save_path_str, epoch))
            print("save weights ! epoch = %d" % epoch)
            segment_net.train()
            pass

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", type=str, default='.', help="root directory of dataset")

    parser.add_argument("--cuda", type=bool, default=False, help="use gpu")
    parser.add_argument("--comet", type=bool, default=True, help="use comet-ml")
    parser.add_argument("--gpu_num", type=int, default=1, help="number of gpu")
    parser.add_argument("--worker_num", type=int, default=4, help="number of input workers")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size of input")
    parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    parser.add_argument("--begin_epoch", type=int, default=0, help="begin_epoch")
    parser.add_argument("--end_epoch", type=int, default=101, help="end_epoch")

    parser.add_argument("--need_test", type=bool, default=True, help="need to test")
    parser.add_argument("--test_interval", type=int, default=1, help="interval of test")
    parser.add_argument("--need_save", type=bool, default=True, help="need to save")
    parser.add_argument("--save_interval", type=int, default=1, help="interval of save weights")

    parser.add_argument("--img_height", type=int, default=360, help="size of image height")
    parser.add_argument("--img_width", type=int, default=640, help="size of image width")

    opt = parser.parse_args()

    print(opt)
    main(opt)
