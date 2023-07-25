import torch
import torch.nn.functional as F
import functools


class MSE_ISSIM(torch.nn.Module):
    @ torch.no_grad()
    def __init__(
            self,
            data_range=1,
            win_size=11,
            win_sigma=1.5,
            channel=1,
            k1=0.0001,
            k2=0.0009,
            issim_win_size=31,
            disable_issim=False,
            normalize_mse=False,
            ssim_coefficient=1.
    ):
        super(MSE_ISSIM, self).__init__()
        self.disable_issim = disable_issim
        self.c1, self.c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
        self.mse_criterion = torch.nn.MSELoss(size_average=False)

        self.register_buffer('gaussian_kernel', self.get_gaussian_kernel2d(win_size, win_sigma))
        self.filter2d = functools.partial(F.conv2d, padding=(win_size - 1) // 2, groups=channel)

        issim_kernel = torch.full(size=(1, 1, issim_win_size, issim_win_size), fill_value=(1 / issim_win_size ** 2))
        self.register_buffer('issim_kernel', issim_kernel)
        self.issim_filter = functools.partial(F.conv2d, padding=(issim_win_size - 1) // 2)

        self.normalize_mse = normalize_mse
        self.ssim_coefficient = ssim_coefficient

    @staticmethod
    def get_gaussian_kernel2d(kernel_size, sigma):
        coords = torch.arange(kernel_size, dtype=torch.float)
        coords -= kernel_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = (gauss / gauss.sum()).unsqueeze(0)
        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        return kernel_2d.view(1, 1, kernel_size, kernel_size)

    def calculate_ssim_map(self, pred: torch.Tensor, target):
        # compute local mean per channel
        mu1 = self.filter2d(pred, self.gaussian_kernel)
        mu2 = self.filter2d(target, self.gaussian_kernel)
        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # compute local sigma per channel
        sigma1_sq = self.filter2d(torch.pow(pred, 2), self.gaussian_kernel) - mu1_sq
        sigma2_sq = self.filter2d(torch.pow(target, 2), self.gaussian_kernel) - mu2_sq
        sigma12 = self.filter2d(pred * target, self.gaussian_kernel) - mu1_mu2

        ssim_map_1 = (2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)
        ssim_map_2 = (mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2)
        return ssim_map_1 / ssim_map_2

    def forward(self, pred, target, points_map):
        mse_loss = self.mse_criterion(pred, target)

        ssim_map = self.calculate_ssim_map(pred, target)
        ssim_loss = 1 - torch.clamp(ssim_map, -1, 1)
        if not self.disable_issim:
            points_map = points_map.to(next(self.named_buffers())[1].device)
            ssim_loss = ssim_loss * self.issim_filter(points_map.unsqueeze(1), self.issim_kernel)
        n_points = max(points_map.sum(), 1)
        ssim_loss = (ssim_loss.sum() / n_points) * self.ssim_coefficient
        if self.normalize_mse:
            mse_loss = mse_loss / n_points
        return mse_loss, ssim_loss
