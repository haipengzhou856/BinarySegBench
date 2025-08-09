import torch
import torch.nn as nn
import torch.nn.functional as F


def get_alpha_gradients(alpha_matte):
    if alpha_matte.ndim == 3:
        alpha_matte_unsqueezed = alpha_matte.unsqueeze(1)
    elif alpha_matte.ndim == 4 and alpha_matte.shape[1] == 1:
        alpha_matte_unsqueezed = alpha_matte


    sobel_x_kernel = torch.tensor([[-1., 0., 1.],
                                   [-2., 0., 2.],
                                   [-1., 0., 1.]], device=alpha_matte.device, dtype=alpha_matte.dtype).reshape(1, 1, 3, 3)
    sobel_y_kernel = torch.tensor([[-1., -2., -1.],
                                   [0., 0., 0.],
                                   [1., 2., 1.]], device=alpha_matte.device, dtype=alpha_matte.dtype).reshape(1, 1, 3, 3)

    padded_alpha = F.pad(alpha_matte_unsqueezed, (1, 1, 1, 1), mode='reflect')
    grad_x = F.conv2d(padded_alpha, sobel_x_kernel, padding='valid')
    grad_y = F.conv2d(padded_alpha, sobel_y_kernel, padding='valid')
    grad_magnitude = torch.abs(grad_x) + torch.abs(grad_y)
    return grad_magnitude

def gradient_loss(pred_alpha, true_alpha, loss_weight=1.0):
    if pred_alpha.ndim == 3: pred_alpha = pred_alpha.unsqueeze(1)
    if true_alpha.ndim == 3: true_alpha = true_alpha.unsqueeze(1)

    pred_alpha_gradients = get_alpha_gradients(pred_alpha)
    true_alpha_gradients = get_alpha_gradients(true_alpha)

    gradient_loss = F.l1_loss(pred_alpha_gradients, true_alpha_gradients)
    return loss_weight * gradient_loss


def composition_loss(pred_alpha, true_alpha, image_rgb, loss_weight=1.0):

    if pred_alpha.ndim == 3: pred_alpha = pred_alpha.unsqueeze(1)
    if true_alpha.ndim == 3: true_alpha = true_alpha.unsqueeze(1)

    if image_rgb.ndim == 3 and image_rgb.shape[0] == 3: # [C, H, W] for single image
        image_rgb = image_rgb.unsqueeze(0) # to [1, C, H, W]
    elif image_rgb.ndim == 4 and image_rgb.shape[3] == 3: # [N, H, W, C]
        image_rgb = image_rgb.permute(0, 3, 1, 2) # to [N, C, H, W]



    composition_pred = pred_alpha * image_rgb
    composition_true = true_alpha * image_rgb

    composition_loss = F.l1_loss(composition_pred, composition_true)
    return loss_weight * composition_loss


def regression_loss(logit, target, loss_type='l2', weight=None):
    """
    Alpha reconstruction loss
    :param logit:
    :param target:
    :param loss_type: "l1" or "l2"
    :param weight: tensor with shape [N,1,H,W] weights for each pixel
    :return:
    """
    if weight is None:
        if loss_type == 'l1':
            return F.l1_loss(logit, target)
        elif loss_type == 'l2':
            return F.mse_loss(logit, target)
        else:
            raise NotImplementedError("NotImplemented loss type {}".format(loss_type))


def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels

def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid

def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel

def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img

def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]