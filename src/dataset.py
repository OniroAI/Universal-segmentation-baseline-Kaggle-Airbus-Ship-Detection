import os
import cv2
import pickle
from torch.utils.data import Dataset

IMG_EXT = '.jpg'
TRG_EXT = '.png'

class ShipDataset(Dataset):
    def __init__(self, img_ids, imgs_dir, trgs_dir=None, masks=False,
                 transform=None, image_transform=None,
                 target_transform=None):
        super().__init__()

        self.masks = masks
        self.img_paths = [os.path.join(imgs_dir, img+IMG_EXT)
                          for img in img_ids]
        if masks:
            self.trg_paths = [os.path.join(trgs_dir, img+TRG_EXT)
                              for img in img_ids]

        self.transform = transform
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        img = cv2.imread(self.img_paths[index])
        if self.masks:
            trg = cv2.imread(self.trg_paths[index], cv2.IMREAD_GRAYSCALE)
            if self.transform is not None:
                img, trg = self.transform(img, trg)
        if self.masks and self.target_transform is not None:
            trg = self.target_transform(trg)
        if self.image_transform is not None:
            img = self.image_transform(img)
        if self.masks:
            return img, trg
        else:
            return img


    def __len__(self):
        return len(self.img_paths)


class ShipDatasetFolds(Dataset):
    def __init__(self, folds_file, folds, imgs_dir, trgs_dir=None, masks=False,
                 transform=None, image_transform=None,
                 target_transform=None):
        super().__init__()
        
        
        with open(folds_file, 'rb') as handle:
            folds_dict = pickle.load(handle)

        img_ids = []
        for k in folds_dict.keys():
            if folds_dict[k] in folds:
                img_ids.append(k)
        print("Use images:", len(img_ids))
        self.masks = masks
        self.img_paths = [os.path.join(imgs_dir, img+IMG_EXT)
                          for img in img_ids]
        print(self.img_paths)
        if masks:
            self.trg_paths = [os.path.join(trgs_dir, img+TRG_EXT)
                              for img in img_ids]

        self.transform = transform
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        img = cv2.imread(self.img_paths[index])
        if self.masks:
            trg = cv2.imread(self.trg_paths[index], cv2.IMREAD_GRAYSCALE)
            if self.transform is not None:
                img, trg = self.transform(img, trg)
        if self.masks and self.target_transform is not None:
            trg = self.target_transform(trg)
        if self.image_transform is not None:
            img = self.image_transform(img)
        if self.masks:
            return img, trg
        else:
            return img


    def __len__(self):
        return len(self.img_paths)
