from torchvision import transforms
import random
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from skimage import color
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
import numbers
import cv2
from scipy import ndimage
import torch


def get_train_joint_transform(scale=(512, 512),
                              mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225],
                              theta=0.03,
                              alpha=2, sigma=0.05):

    random_choise = random.choices([
                    RandomRotate(),
                    RandomAffineCV2(0.1),
                    HEDJitter(theta=theta),
                    RandomElastic(alpha=alpha, sigma=sigma),
                    RandomGaussBlur(radius=[0.5, 1.5]),
                    transforms.ColorJitter(hue=0.5)],
                    k=3) # select 3 ways

    joint_transform = Compose([
        random_choise[0],
        random_choise[1],
        random_choise[2],
        RandomFlip(),
        RandomCropAndResize(scale),
        ToTensor(),
        Normalize(mean, std)
    ])
    return joint_transform



def get_test_joint_transform(scale=(512, 512)):
    joint_transform = Compose([
        Resize(scale),
        ToTensor(),
    ])
    return joint_transform


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        return F.center_crop(img, self.size), F.center_crop(mask, self.size)


class RandomCrop(object):
    def __init__(self, ratio=(0.0546875, 0.03125)):
        # ratio[0] is for short edge, ratio[1] is for long edge
        self.ratio = ratio

    def __call__(self, img, mask):
        w, h = img.size
        if w > h:
            self.ratio = (self.ratio[1], self.ratio[0])

        tw, th = int(w * (1 - self.ratio[0])), int(h * (1 - self.ratio[1]))

        x, y = random.randint(0, w - tw), random.randint(0, h - th)

        img, mask = img.crop((x, y, x + tw, y + th)), mask.crop((x, y, x + tw, y + th))

        return img, mask


class RandomCropAndResize(object):
    def __init__(self, scale=(256, 488)):
        assert scale[0] <= scale[1]
        self.scale = scale
        self.crop = RandomCrop(ratio=(0.1, 0.06))

    def __call__(self, img, mask):
        img, mask = self.crop(img, mask)
        w, h = img.size

        if w > h:
            img, mask = (img.resize((self.scale[1], self.scale[0]), Image.BILINEAR),
                         mask.resize((self.scale[1], self.scale[0]), Image.NEAREST))
        else:
            img, mask = (img.resize((self.scale[0], self.scale[1]), Image.BILINEAR),
                         mask.resize((self.scale[0], self.scale[1]), Image.NEAREST))
        return img, mask


class RandomFlip(object):
    def __call__(self, img, mask):
        horizontal_flip = random.random() < 0.5
        vertical_flip = random.random() < 0.5

        if horizontal_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if vertical_flip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return img, mask


class HEDJitter(object):
    def __init__(self, theta=0.):  # HED_light: theta=0.05; HED_strong: theta=0.2
        self.theta = theta
        self.alpha = np.random.uniform(1 - theta, 1 + theta, (1, 3))
        self.betti = np.random.uniform(-theta, theta, (1, 3))

    @staticmethod
    def adjust_HED(img, alpha, betti):
        img = np.array(img)

        s = np.reshape(color.rgb2hed(img), (-1, 3))
        ns = alpha * s + betti  # perturbations on HED color space
        nimg = color.hed2rgb(np.reshape(ns, img.shape))

        imin = nimg.min()
        imax = nimg.max()
        rsimg = (255 * (nimg - imin) / (imax - imin)).astype('uint8')  # rescale to [0,255]
        # transfer to PIL image
        return Image.fromarray(rsimg)

    def __call__(self, img, mask):
        return self.adjust_HED(img, self.alpha, self.betti), mask

# avoid use it, this augmentation is too slow
class RandomElastic(object):
    def __init__(self, alpha, sigma):
        assert 0.05 <= sigma <= 0.1, \
            "In pathological image, sigma should be in (0.05,0.1)"
        self.alpha = alpha
        self.sigma = sigma

    @staticmethod
    def RandomElasticCV2(img, alpha, sigma, mask=None):
        alpha = img.shape[1] * alpha
        sigma = img.shape[1] * sigma
        if mask is not None:
            mask = np.array(mask).astype(np.uint8)
            img = np.concatenate((img, mask[..., None]), axis=2)

        shape = img.shape

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        # dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        img = map_coordinates(img, indices, order=0, mode='reflect').reshape(shape)
        if mask is not None:
            return Image.fromarray(img[..., :3]), Image.fromarray(img[..., 3])
        else:
            return Image.fromarray(img)

    def __call__(self, img, mask=None):
        return self.RandomElasticCV2(np.array(img), self.alpha, self.sigma, mask)


class RandomAffineCV2(object):
    """Random Affine transformation by CV2 method on image by alpha parameter.
    Args:
        alpha (float): alpha value for affine transformation
        mask (PIL Image) in __call__, if not assign, set None.
    """

    def __init__(self, alpha):
        assert isinstance(alpha, numbers.Number), "alpha should be a single number."
        assert 0. <= alpha <= 0.15, \
            "In pathological image, alpha should be in (0,0.15), you can change in myTransform.py"
        self.alpha = alpha

    @staticmethod
    def affineTransformCV2(img, alpha, mask=None):
        alpha = img.shape[1] * alpha
        if mask is not None:
            mask = np.array(mask).astype(np.uint8)
            img = np.concatenate((img, mask[..., None]), axis=2)

        imgsize = img.shape[:2]
        center = np.float32(imgsize) // 2
        censize = min(imgsize) // 3
        pts1 = np.float32([center + censize, [center[0] + censize, center[1] - censize], center - censize])  # raw point
        pts2 = pts1 + np.random.uniform(-alpha, alpha, size=pts1.shape).astype(np.float32)  # output point
        M = cv2.getAffineTransform(pts1, pts2)  # affine matrix
        img = cv2.warpAffine(img, M, imgsize[::-1],
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        if mask is not None:
            return Image.fromarray(img[..., :3]), Image.fromarray(img[..., 3])
        else:
            return Image.fromarray(img)

    def __call__(self, img, mask=None):
        return self.affineTransformCV2(np.array(img), self.alpha, mask)

    def __repr__(self):
        return self.__class__.__name__ + '(alpha value={0})'.format(self.alpha)


class RandomGaussBlur(object):
    def __init__(self, radius=None):
        if radius is not None:
            assert isinstance(radius, (tuple, list)) and len(radius) == 2, \
                "radius should be a list or tuple and it must be of length 2."
            self.radius = random.uniform(radius[0], radius[1])
        else:
            self.radius = 0.0

    def __call__(self, img, mask):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius)), mask


class Resize(object):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, mask):
        img = F.resize(img, self.size, self.interpolation)
        mask = F.resize(mask, self.size, self.interpolation)
        return img, mask

class RandomRotate(object):
    def __init__(self, angle_range=(-15, 15)):
        self.angle_range = angle_range

    def __call__(self, img, mask):
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        img = img.rotate(angle, resample=Image.BILINEAR)
        mask = mask.rotate(angle, resample=Image.NEAREST)

        return img, mask
class ToTensor(object):
    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, img, mask):
        img = self.totensor(img)
        mask = self.totensor(mask)

        return img, mask


class Normalize(object):
    def __init__(self, mean, std):
        self.normlize = transforms.Normalize(mean, std)

    def __call__(self, img, mask):
        img = self.normlize(img)
        return img, mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils.utils import reverse_normalize, Tensor2PIL
    from torch.utils.tensorboard import SummaryWriter




    img = "/home/haipeng/Code/Data/BIGSHA/test/img/DIS_0011.jpg"
    mask = "/home/haipeng/Code/Data/BIGSHA/test/mask/DIS_0011.png"

    img = Image.open(img).convert('RGB')
    mask = Image.open(mask).convert('L')

    scale = (512, 512)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    aug = Compose([
        #Resize(scale),
        RandomAffineCV2(0.1),
        #RandomRotate(),
        RandomGaussBlur(radius=[0.1,5]),
        #RandomCropAndResize(scale),
        #transforms.ColorJitter(hue=0.5),
        #RandomFlip(),
        #HEDJitter(theta=0.03),
        #RandomElastic(alpha=2, sigma=0.05),
        ToTensor(),
        Normalize(mean, std)
    ])

    aug_img, aug_mask = aug(img, mask)

    aug_img = reverse_normalize(aug_img)
    aug_img = Tensor2PIL(aug_img)
    aug_mask = Tensor2PIL(aug_mask)


    img = np.array(img)
    mask = np.array(mask)

    aug_img = np.array(aug_img)
    aug_mask = np.array(aug_mask)

    fig, axes = plt.subplots(2, 2)

    # Display img1 in the first subplot
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Image 1')

    # Display img2 in the second subplot
    axes[0, 1].imshow(mask)
    axes[0, 1].set_title('Image 2')

    # Display img3 in the third subplot
    axes[1, 0].imshow(aug_img)
    axes[1, 0].set_title('Image 3')

    # Display img4 in the fourth subplot
    axes[1, 1].imshow(aug_mask)
    axes[1, 1].set_title('Image 4')

    # Adjust the layout of the subplots
    plt.tight_layout()

    # Show the figure
    plt.show()


