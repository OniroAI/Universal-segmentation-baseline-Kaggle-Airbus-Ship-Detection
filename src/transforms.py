import torch
import random
import numpy as np
import cv2
import collections

from src.target import MakeTarget

class ImageToTensor:
    def __call__(self, image):
        image = np.moveaxis(image, -1, 0)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        return image


class MaskToTensor:
    def __init__(self):
        self.make_target = MakeTarget()
    
    def __call__(self, mask):
        inp = self.make_target(mask)
        return torch.from_numpy(inp)


def img_crop(img, box):
    return img[box[1]:box[3], box[0]:box[2]]


def random_crop(img, size):
    tw = size[0]
    th = size[1]
    w, h = img_size(img)
    if ((w - tw) > 0) and ((h - th) > 0):
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
    else:
        x1 = 0
        y1 = 0
    img_return = img_crop(img, (x1, y1, x1 + tw, y1 + th))
    return img_return, x1, y1


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask=None):
        w, h = img.shape[1], img.shape[0]
        tw, th = self.size
        if w == tw and h == th:
            if mask is None:
                return img
            else:
                return img, mask

        x1 = (w - tw) // 2
        y1 = (h - th) // 2

        crop_img = img_crop(img, (x1, y1, x1 + tw, y1 + th))

        if mask is None:
            return crop_img
        crop_mask = img_crop(mask, (x1, y1, x1 + tw, y1 + th))
        return crop_img, crop_mask


class ProbOutputTransform:
    def __init__(self, segm_thresh=0.5, prob_thresh=0.0001, size=None):
        self.segm_thresh = segm_thresh
        self.prob_thresh = prob_thresh
        self.crop = None
        if size is not None:
            self.crop = CenterCrop(size)

    def __call__(self, preds):
        segms, probs = preds
        preds = segms > self.segm_thresh
        probs = probs > self.prob_thresh
        preds = preds * probs.view(-1, 1, 1, 1)
        if self.crop is not None:
            preds = self.crop(preds)
        return preds


def img_size(image: np.ndarray):
    """
    Return images width and height.
    :param image: nd.array with image
    :return: width, height
    """
    return (image.shape[1], image.shape[0])


class Scale:
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = tuple(size)
        self.interpolation = interpolation

    def __call__(self, image, mask=None):
        resize_image = cv2.resize(image, self.size, interpolation=self.interpolation)
        if mask is None:
            return resize_image
        resize_mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        return resize_image, resize_mask


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        img, x1, y1 = random_crop(img, self.size)
        if mask is None:
            return img
        mask = img_crop(mask, (x1, y1, x1 + self.size[0],
                               y1 + self.size[1]))
        return img, mask

class RandomScaleCrop:
    def __init__(self, size, prob, min_size=256, max_size=768):
        self.scaler = Scale(size)
        self.min_size = min_size
        self.max_size = max_size
        self.prob = prob

    def __call__(self, img, mask):
        img_crop, mask_crop = self._get_crop(img, mask)
        if mask is None:
            img_crop = self.scaler(img_crop)
            return img_crop
        while(np.count_nonzero(mask_crop) == 0\
              and np.random.rand() < self.prob):
            img_crop, mask_crop = self._get_crop(img, mask)
        img_crop, mask_crop = self.scaler(img_crop, mask_crop)
        return img_crop, mask_crop
    
    def _get_crop(self, img, mask):
        w = np.random.randint(self.min_size, self.max_size)
        img_c, x1, y1 = random_crop(img, (w, w))
        if mask is not None:
            mask_c = img_crop(mask, (x1, y1, x1 + w, y1 + w))
        return img_c, mask_c


class RandomCropNotEmptyProb:
    def __init__(self, size, prob=0.5):
        self.prob = prob
        self.croper = RandomCrop(size)

    def __call__(self, img, mask):
        img_crop, mask_crop = self.croper(img, mask)
        while(np.count_nonzero(mask_crop) == 0\
              and np.random.rand() < self.prob):
            img_crop, mask_crop = self.croper(img, mask)
        return img_crop, mask_crop


class Flip:
    def __init__(self, flip_code):
        assert flip_code in [0, 1]
        self.flip_code = flip_code

    def __call__(self, img, mask=None):
        img = cv2.flip(img, self.flip_code)
        if mask is None:
            return img
        mask = cv2.flip(mask, self.flip_code)
        return img, mask


class GaussNoise:
    def __init__(self, sigma_sq):
        self.sigma_sq = sigma_sq

    def __call__(self, img, mask):
        if self.sigma_sq > 0.0:
            img = self._gauss_noise(img,
                    np.random.uniform(0, self.sigma_sq))
        return img, mask

    def _gauss_noise(self, img, sigma_sq):
        img = img.astype(np.int32)
        w, h, c = img.shape
        gauss = np.random.normal(0, sigma_sq, (h, w))
        gauss = gauss.reshape(h, w)
        img = img + np.stack([gauss for i in range(c)], axis=2)
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        return img


class SpeckleNoise:
    def __init__(self, sigma_sq):
        self.sigma_sq = sigma_sq

    def __call__(self, img, mask):
        if self.sigma_sq > 0.0:
            img = self._speckle_noise(img,
                    np.random.uniform(0, self.sigma_sq))
        return img, mask

    def _speckle_noise(self, img, sigma_sq):
        sigma_sq /= 255
        img = img.astype(np.int32)
        w, h, c = img.shape
        gauss = np.random.normal(0, sigma_sq, (h, w))
        gauss = gauss.reshape(h, w)
        img = img + np.stack([gauss for i in range(c)], axis=2) * img
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        return img


class AugmentImage(object):
    def __init__(self, augment_parameters):
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2

    def __call__(self, img, trgs_mask=None):
        p = np.random.uniform(0, 1, 1)
        random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
        random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
        random_colors = np.random.uniform(self.color_low, self.color_high, 3)
        img = img
        # randomly shift gamma
        img = img ** random_gamma
        # randomly shift brightness
        img = img * random_brightness
        # randomly shift color
        for i in range(3):
            img[:, :, i] *= random_colors[i]
        # saturate
        img = np.clip(img, 0, 255)

        return img, trgs_mask



class RandomRotate(object):
    def __init__(self, max_ang=0):
        '''Random image rotate around the image center
        Args:
            max_ang (float): Max angle of rotation in deg
        '''
        self.max_ang = max_ang

    def __call__(self, img, mask=None):
        if self.max_ang != 0:
            h, w, _ = img.shape

            ang = np.random.uniform(-self.max_ang, self.max_ang)
            M = cv2.getRotationMatrix2D((w/2,h/2), ang, 1)
            img = cv2.warpAffine(img, M, (w, h))

            if mask is not None:
                mask = cv2.warpAffine(mask, M, (w, h))
                return img, mask
            return img
        else:
            return img, mask


class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, mask=None):
        if mask is None:
            if random.random() < self.prob:
                image = self.transform(image)
            return image
        else:
            if random.random() < self.prob:
                image, mask = self.transform(image, mask)
            return image, mask

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        if mask is None:
            for trns in self.transforms:
                image = trns(image)
            return image
        else:
            for trns in self.transforms:
                image, mask = trns(image, mask)
            return image, mask


def train_transforms(size=(256, 256), skip_empty_prob=0.5, sigma_g=10):
    transforms_dict = dict()
    transforms_dict['transform'] = Compose([
        UseWithProb(RandomRotate(20), 0.05),
        #RandomCropNotEmptyProb(size, skip_empty_prob),
        RandomScaleCrop(size, skip_empty_prob),
        UseWithProb(AugmentImage([0.95, 1.05, 0.9, 1.12, 0.95, 1.05]), 0.1),
        UseWithProb(GaussNoise(sigma_g), 0.1),
        UseWithProb(Flip(0), 0.25),
        UseWithProb(Flip(1), 0.25),
        #UseWithProb(RandomRotate(20), 0.1)
    ])

    transforms_dict['image_transform'] = ImageToTensor()
    transforms_dict['target_transform'] = MaskToTensor()
    return transforms_dict


def test_transforms(size=(256, 256)):
    transforms_dict = dict()
    transforms_dict['transform'] = Compose([
        CenterCrop(size),
    ])

    transforms_dict['image_transform'] = ImageToTensor()
    transforms_dict['target_transform'] = MaskToTensor()
    return transforms_dict
