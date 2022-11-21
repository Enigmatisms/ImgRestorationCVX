import torch
import tqdm
import random
import numpy as np
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.io
import argparse
import cv2 as cv
from torchvision.utils import save_image

def make_laplace():
    return torch.FloatTensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

def make_conv_kernel(single_layer: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((3, 3, single_layer.shape[0], single_layer.shape[1]), device = single_layer.device)
    result[0, 0] = single_layer.clone()
    result[1, 1] = single_layer.clone()
    result[2, 2] = single_layer.clone()
    return result

class RestoredImage(nn.Module):
    def __init__(self, width = 720, height = 480):
        super().__init__()
        G_mat = scipy.io.loadmat('G.mat')['G']
        print("Gaussian kernel loaded")
        self.conv_kernel = make_conv_kernel(torch.from_numpy(G_mat)).cuda()
        self.conv_kernel_size = 25
        self.w = width
        self.h = height
        self.img = nn.Parameter(torch.rand(1, 3, self.h, self.w), requires_grad = True)
        self.noise = nn.Parameter(torch.zeros(1, 1, self.h, self.w), requires_grad = True)
        self.laplace_kernel = make_conv_kernel(make_laplace()).cuda()
        self.noise_patch_size = 51
        self.use_tv = False
        self.use_noise = True
        
    def matrix_norm_reg(self):
        # random patch matrix spectrum: linear
        half_size = self.noise_patch_size >> 1
        small_eigs = torch.zeros(half_size, device = 'cuda')
        inside = []
        for _ in range(10):             # round 7.7%
            while True:
                anchor_x = random.randint(half_size, self.w - half_size - 1)
                anchor_y = random.randint(half_size, self.h - half_size - 1)
                for item in inside:
                    if abs(anchor_x - item[0]) <= self.noise_patch_size and abs(anchor_y - item[1]) <= self.noise_patch_size:
                        break
                else:
                    inside.append((anchor_x, anchor_y))
                    break
            for channel_id in range(3):
                patch = self.img[-1, channel_id, anchor_y - half_size:anchor_y + half_size + 1, anchor_x - half_size:anchor_x + half_size + 1]
                decentered_patch = patch - torch.mean(patch)                          # decentralize
                _, eigs_sorted, _ = torch.svd(decentered_patch)      # make symmetric
                small_eigs = small_eigs + eigs_sorted[-half_size:] ** 2
        return torch.mean(small_eigs)
        
    @staticmethod 
    def conv(img: torch.Tensor, kernel: torch.Tensor, aux: torch.Tensor = None) -> torch.Tensor:
        pad_size = 25 >> 1
        img = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode = 'replicate')
        if aux is not None:
            return F.conv2d(img, kernel + aux)
        else:
            return F.conv2d(img, kernel)
    
    def calc_tv_loss(self, l1 = True):
        diff_image_y = self.img[:, :-1, :] - self.img[:, 1:, :]
        diff_image_x = self.img[:, :, :-1] - self.img[:, :, 1:]
        if l1 == True:
            return torch.mean(diff_image_y.abs()) + torch.mean(diff_image_x.abs())
        else:
            return torch.mean(diff_image_y ** 2) + torch.mean(diff_image_x ** 2)
        
    def forward(self):
        conv_output = RestoredImage.conv(self.img, self.conv_kernel)
        if self.use_noise:
            conv_output = conv_output + self.noise
        if self.use_tv:
            return conv_output, self.calc_tv_loss()
        else:
            return conv_output, (RestoredImage.conv(self.img, self.laplace_kernel) ** 2).mean()
    
def train_main():
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 2000, help = "Training lasts for . epochs")
    parser.add_argument("--eval_time", type = int, default = 20, help = "Tensorboard output interval (train time)")
    parser.add_argument("--reg_coeff", type = float, default = 0.01, help = "Weight for regularizer norm")
    parser.add_argument("--decay_rate", type = float, default = 0.998, help = "After <decay step>, lr = lr * <decay_rate>")
    parser.add_argument("--lr", type = float, default = 0.2, help = "Start lr")
    parser.add_argument("--name", type = str, default = "./input/blurred3.png", help = "Image name")
    args = parser.parse_args()
    
    epochs              = args.epochs
    eval_time           = args.eval_time
    img = plt.imread(args.name)
    if img.shape[-1] > 3:
        img = img[..., :-1]
    img = torch.from_numpy(img).permute(2, 0, 1).cuda().unsqueeze(0)
    
    rimg = RestoredImage(img.shape[3], img.shape[2]).cuda()
    l2_loss = lambda x, y: torch.mean((x - y) ** 2)
    opt = optim.Adam(rimg.parameters(), lr = args.lr)
    sch = optim.lr_scheduler.ExponentialLR(opt, args.decay_rate)
    
    for ep in tqdm.tqdm(range(0, epochs)):
        result, reg_loss = rimg.forward()
        img_loss = l2_loss(result, img)
        
        loss: torch.Tensor = reg_loss * args.reg_coeff + img_loss
        if rimg.use_noise:
            loss = loss + 1e-4 * rimg.matrix_norm_reg()
    
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(rimg.noise, 1e-4)
        opt.step()
        sch.step()

        if ep % eval_time and ep > 0:
            lr = sch.get_last_lr()[-1]
            print("Traning Epoch: %4d / %4d\ttrain loss: %.4f\timg loss: %.5lf\tl1 norm: %.5lf\tlr:%.7lf"%(
                    ep, epochs, loss.item(), img_loss, reg_loss, lr
            ))
            
    save_image(rimg.img, "tensor.png")
    save_image(rimg.noise, "noise.png")
    print("Output completed.")
    
def pure_conv():
    img = plt.imread("./input/figure3.png")

    G_mat = scipy.io.loadmat('G.mat')['G']
    conv_kernel = make_conv_kernel(torch.from_numpy(G_mat)).cuda()
    img_out = RestoredImage.conv(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda(), conv_kernel)
    save_image(img_out.clamp(0, 1), "conved.png", 1)
    
def min_max_filtering(img:np.ndarray, radius: int = 2, is_max = True):
    if img.ndim < 3:
        img = img.reshape(*img.shape, 1)
    last_dim = img.shape[-1]
    img = torch.from_numpy(img)
    padded_img = F.pad(img, [0, 0, radius, radius, radius, radius], 'constant', 0)
    print(padded_img.shape)
    max_h, max_w = padded_img.shape[0] - radius, padded_img.shape[1] - radius
    output = img.clone()
    for i in tqdm.tqdm(range(radius, max_h)):
        for j in range(radius, max_w):
            center_value = padded_img[i, j, :].norm()
            rgb_values = padded_img[i - radius:i + radius + 1, j - radius:j + radius + 1, :].reshape(-1, last_dim)
            norm_values = rgb_values.norm(dim = -1)
            if is_max == True:
                norm = torch.max(norm_values)
            else:
                norm = torch.min(norm_values)
            # print(f"(i, j) max_norm = {max_norm.item()}, norm_value = {norm_values.item()}")
            if (norm - center_value).abs() < 3e-3:      # 就是最大值
                center_id = ((2 * radius + 1) ** 2) // 2 + 1
                _, idx = torch.kthvalue(norm_values, center_id)
                output[i - radius, j - radius, :] = rgb_values[idx, :]        # 最大值中值化
    return output

def bidirectional_filtering():
    img = plt.imread("./input/blurred3.png")
    print(img.dtype, img.shape)
    # laplace = cv.bilateralFilter(img, 5, 50, 5)
    new_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    new_img[..., 0] = min_max_filtering(new_img[..., 0], 2, False)[..., 0]
    laplace = cv.Laplacian(new_img[..., 0], cv.CV_32F)
    # plt.subplot(2, 1, 1)
    plt.imshow(laplace)
    # plt.subplot(2, 1, 2)
    # plt.imshow(new_img[..., 1])
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # pure_conv()
    # max_filtering()
    # bidirectional_filtering()
    train_main()
    