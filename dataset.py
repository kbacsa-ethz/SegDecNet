import glob
import os
import numpy as np
import cv2

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import torchvision.transforms.functional as VF


class KolektorDataset(Dataset):
    def __init__(self, dataRoot, transforms_=None, transforms_mask=None, subFold="Train_NG", isTrain=True):

        self.isTrain = isTrain
        if transforms_mask == None:
            self.maskTransform = transforms.Compose([transforms.ToTensor()])
        else:
            self.maskTransform = transforms_mask

        if transforms_ == None:
            self.transform = self.maskTransform
        else:
            self.transform = transforms_

        self.imgFiles = sorted(glob.glob(os.path.join(dataRoot, subFold) + "/*.bmp"))

        if isTrain:
            self.labelFiles = sorted(glob.glob(os.path.join(dataRoot, subFold) + "/*.bmp"))

        self.len = len(self.imgFiles)

    def __getitem__(self, index):

        idx = index % self.len

        if self.isTrain == True:

            img = Image.open(self.imgFiles[idx]).convert("RGB")

            # mask = Image.open(self.labelFiles[idx]).convert("RGB")
            mat = cv2.imread(self.labelFiles[idx], cv2.IMREAD_GRAYSCALE)
            kernel = np.ones((5, 5), np.uint8)
            matD = cv2.dilate(mat, kernel)
            mask = Image.fromarray(matD)  # image2 is a PIL image

            if np.random.rand(1) > 0.5:
                mask = VF.hflip(mask)
                img = VF.hflip(img)

            if np.random.rand(1) > 0.5:
                mask = VF.vflip(mask)
                img = VF.vflip(img)

            img = self.transform(img)
            mask = self.maskTransform(mask)

            return {"img": img, "mask": mask}
        else:
            img = Image.open(self.imgFiles[idx]).convert("RGB")
            img = self.transform(img)
            return {"img": img}

    def __len__(self):
        return len(self.imgFiles)


class TokaidoTextureDataset(Dataset):
    def __init__(self, dataRoot, transforms_=None, transforms_mask=None, dataFrame=None, isTrain=True):

        self.dataRoot = dataRoot
        self.isTrain = isTrain
        if transforms_mask == None:
            self.maskTransform = transforms.Compose([transforms.ToTensor()])
        else:
            self.maskTransform = transforms_mask

        if transforms_ == None:
            self.transform = self.maskTransform
        else:
            self.transform = transforms_

        self.imgFiles = dataFrame
        self.len = len(self.imgFiles)

    def __getitem__(self, index):

        idx = index % self.len

        if self.isTrain == True:

            img = Image.open(os.path.join(self.dataRoot, self.imgFiles['image_path'].iloc[idx])).convert("RGB")

            # mask = Image.open(self.labelFiles[idx]).convert("RGB")
            mat = cv2.imread(os.path.join(self.dataRoot, self.imgFiles['damage_path'][idx]), cv2.IMREAD_GRAYSCALE)
            kernel = np.ones((5, 5), np.uint8)
            matD = cv2.dilate(mat, kernel)
            mask = Image.fromarray(matD)  # image2 is a PIL image

            if np.random.rand(1) > 0.5:
                mask = VF.hflip(mask)
                img = VF.hflip(img)

            if np.random.rand(1) > 0.5:
                mask = VF.vflip(mask)
                img = VF.vflip(img)

            img = self.transform(img)
            mask = self.maskTransform(mask)

            return img, mask
        else:
            img = Image.open(os.path.join(self.dataRoot, self.imgFiles['image_path'].iloc[idx])).convert("RGB")
            img = self.transform(img)
            return img

    def __len__(self):
        return len(self.imgFiles)


if __name__ == '__main__':
    import pandas as pd

    root_dir = '/home/kiran/Tokaido_dataset'

    df_puretex_train = pd.read_csv(os.path.join(root_dir, 'files_puretex_test.csv'),
                                   names=['image_path', 'damage_path'])
    df_puretex_train = df_puretex_train.replace(to_replace=r'\\', value='/', regex=True)

    dataset = TokaidoTextureDataset(
        root_dir,
        transforms_=None,
        transforms_mask=None,
        dataFrame=df_puretex_train,
        isTrain=False
    )

    img = next(iter(dataset))
    print(img)
