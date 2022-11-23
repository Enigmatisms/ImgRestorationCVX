import tqdm
import torch
import random
import scipy.io
import argparse
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast as autocast
from torch.nn.utils.clip_grad import clip_grad_norm_
from utils import make_conv_kernel, make_laplace, make_scharr

class RestoredImage(nn.Module):
    def __init__(self, width = 720, height = 480, use_cuda = True, est_noise = True, use_scharr = True):
        super().__init__()
        G_mat = scipy.io.loadmat('G.mat')['G']
        self.device = 'cuda' if use_cuda else 'cpu'
        self.conv_kernel = make_conv_kernel(torch.from_numpy(G_mat).to(self.device))
        self.conv_kernel_size = 25
        self.w = width
        self.h = height
        self.img = nn.Parameter(torch.rand(1, 3, self.h, self.w, device = self.device), requires_grad = True)
        self.noise = nn.Parameter(torch.zeros(1, 1, self.h, self.w, device = self.device), requires_grad = True)
        self.laplace_kernel = make_conv_kernel(make_laplace()).to(self.device)
        self.noise_patch_size = 31
        self.use_scharr = use_scharr
        if self.use_scharr:
            self.scharrs = [make_conv_kernel(make_scharr(i)).to(self.device) for i in range(6)]
            self.scharrs.append(make_conv_kernel(make_laplace()).to(self.device))
        self.use_noise = est_noise
        
    def matrix_norm_reg(self):
        half_size = self.noise_patch_size >> 1
        diff_eigs = torch.zeros(self.noise_patch_size - 3, device = 'cuda')
        inside = []
        for _ in range(16):             # round 7.7%
            while True:
                anchor_x = random.randint(half_size, self.w - half_size - 1)
                anchor_y = random.randint(half_size, self.h - half_size - 1)
                for item in inside:
                    if abs(anchor_x - item[0]) <= self.noise_patch_size and abs(anchor_y - item[1]) <= self.noise_patch_size: break
                else:
                    inside.append((anchor_x, anchor_y))
                    break
            patch = self.noise[-1, -1, anchor_y - half_size:anchor_y + half_size + 1, anchor_x - half_size:anchor_x + half_size + 1]
            decentered_patch = patch - torch.mean(patch)                          # decentralize
            _, eigs_sorted, _ = torch.svd(decentered_patch)      # make symmetric
            eigs_sorted = eigs_sorted[1:]
            diff1 = eigs_sorted[:-1] - eigs_sorted[1:]
            diff2 = diff1[:-1] - diff1[1:]
            diff_eigs = diff_eigs + diff2 ** 2
        return torch.mean(diff_eigs)

    def min_spectrum_tail(self):
        tail_size = 30
        patch = self.img.squeeze(0).mean(dim = 0)
        decentered_patch = patch - torch.mean(patch)                          # decentralize
        _, eigs_sorted, _ = torch.svd(decentered_patch)      # make symmetric
        return torch.mean(eigs_sorted[-tail_size:] ** 2)
        
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
        diff_image_d = self.img[:, 1:, 1:] - self.img[:, :-1, :-1]
        if l1 == True:
            return torch.mean(diff_image_y.abs()) + torch.mean(diff_image_x.abs())
        else:
            return torch.mean(diff_image_y ** 2) + torch.mean(diff_image_x ** 2) + \
                   torch.mean(diff_image_d ** 2)
        
    def forward(self):
        conv_output = RestoredImage.conv(self.img, self.conv_kernel)
        if self.use_noise:
            conv_output = conv_output + self.noise
        if self.use_scharr:
            first_order = sum([(RestoredImage.conv(self.img, kernel) ** 2).mean() for kernel in self.scharrs])
            return conv_output, first_order
        else:
            return conv_output, (RestoredImage.conv(self.img, self.laplace_kernel) ** 2).mean()
    
def train_main():
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 150, help = "Training lasts for . epochs")
    parser.add_argument("--eval_time", type = int, default = 20, help = "Tensorboard output interval (train time)")
    parser.add_argument("--reg_coeff", type = float, default = 0.0004, help = "Weight for regularizer norm")
    parser.add_argument("--decay_rate", type = float, default = 0.998, help = "After <decay step>, lr = lr * <decay_rate>")
    parser.add_argument("-o", "--optimize", action = 'store_true', default = False, help = "Use auto-mixed precision optimization")
    parser.add_argument("--cpu", action = 'store_true', default = False, help = "Disable GPU acceleration")
    parser.add_argument("--scharr", action = 'store_true', default = False, help = "Use first order gradient Scharr kernel")
    parser.add_argument("-n", "--est_noise", action = 'store_true', default = False, help = "Whether to estimate noise")
    parser.add_argument("--lr", type = float, default = 0.2, help = "Start lr")
    parser.add_argument("--name", type = str, default = "./input/blurred3.png", help = "Image name")
    args = parser.parse_args()
    
    epochs              = args.epochs
    eval_time           = args.eval_time
    use_cuda            = (not args.cpu) and torch.cuda.is_available()
    use_amp             = args.optimize
    est_noise           = args.est_noise  
    use_scharr          = args.scharr
    device = 'cuda' if use_cuda else 'cpu'

    img = plt.imread(args.name)
    if img.shape[-1] > 3:
        img = img[..., :-1]
    img = torch.from_numpy(img).permute(2, 0, 1).to(device).unsqueeze(0)
    
    rimg = RestoredImage(img.shape[3], img.shape[2], use_cuda = use_cuda, est_noise = est_noise, use_scharr = use_scharr).to(device)
    l2_loss = lambda x, y: torch.mean((x - y) ** 2)
    opt = optim.Adam(rimg.parameters(), lr = args.lr)
    sch = optim.lr_scheduler.ExponentialLR(opt, args.decay_rate)

    if use_amp:
        scaler = GradScaler()
    else:
        scaler = None

    def train_epoch():
        result, reg_loss = rimg.forward()
        img_loss = l2_loss(result, img)
        loss: torch.Tensor = reg_loss * args.reg_coeff + img_loss
        if rimg.use_noise:
            loss = loss + 1e-4 * rimg.min_spectrum_tail()
        return loss, img_loss, reg_loss
    for ep in tqdm.tqdm(range(0, epochs)):

        loss, img_loss, reg_loss = train_epoch()
        if use_amp:
            with autocast():
                scaler.scale(loss).backward()
                if est_noise:
                    clip_grad_norm_(rimg.noise, 4e-5)
                scaler.step(opt)
                scaler.update()
        else:
            opt.zero_grad()
            loss.backward()
            if est_noise:
                clip_grad_norm_(rimg.noise, 4e-5)
            opt.step()
        sch.step()

        if ep % eval_time == 0 and ep > 0:
            lr = sch.get_last_lr()[-1]
            print("Traning Epoch: %4d / %4d\ttrain loss: %.4f\timg loss: %.5lf\tl1 norm: %.5lf\tlr:%.7lf"%(
                    ep, epochs, loss.item(), img_loss, reg_loss, lr
            ))
            
    save_image(rimg.img, "tensor.png")
    if est_noise:
        save_image(rimg.noise, "noise.png")
    print("Output completed.")

if __name__ == "__main__":
    train_main()
    