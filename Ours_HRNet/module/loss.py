import torch
import torch.nn.functional as F


def smoothness_loss(sal, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x))
    weight_y = torch.exp(-torch.abs(img_grad_y))

    loss = (((sal[:, :, :, :-1] - sal[:, :, :, 1:]).abs() * weight_x).sum() + \
           ((sal[:, :, :-1, :] - sal[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())

    return loss


def laplacian_edge(img):
    laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filter = torch.reshape(laplacian_filter, [1, 1, 3, 3])
    filter = filter.cuda()
    lap_edge = F.conv2d(img, filter, stride=1, padding=1)
    return lap_edge


def gradient_x(img):
    sobel = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter = torch.reshape(sobel,[1,1,3,3])
    filter = filter.cuda()
    gx = F.conv2d(img, filter, stride=1, padding=1)
    return gx


def gradient_y(img):
    sobel = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter = torch.reshape(sobel, [1, 1,3,3])
    filter = filter.cuda()
    gy = F.conv2d(img, filter, stride=1, padding=1)
    return gy


def charbonnier_penalty(s):
    cp_s = torch.pow(torch.pow(s, 2) + 0.001**2, 0.5)
    return cp_s


def smoothness_loss_jing(sal, img):
    alpha = 10
    s1 = 10
    s2 = 1
    ## first oder derivative: sobel
    sal_x = torch.abs(gradient_x(sal))
    sal_y = torch.abs(gradient_y(sal))
    gt_x = gradient_x(img)
    gt_y = gradient_y(img)
    w_x = torch.exp(torch.abs(gt_x) * (-alpha))
    w_y = torch.exp(torch.abs(gt_y) * (-alpha))
    cps_x = charbonnier_penalty(sal_x * w_x)
    cps_y = charbonnier_penalty(sal_y * w_y)
    cps_xy = cps_x + cps_y

    ## second order derivative: laplacian
    lap_sal = torch.abs(laplacian_edge(sal))
    lap_gt = torch.abs(laplacian_edge(img))
    weight_lap = torch.exp(lap_gt * (-alpha))
    weighted_lap = charbonnier_penalty(lap_sal*weight_lap)

    smooth_loss = s1*torch.mean(cps_xy) + s2*torch.mean(weighted_lap)

    return smooth_loss
