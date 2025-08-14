import torch
# Loss functions
def si_sdr_loss(pred, target, eps=1e-8):
    """
    SI_SDRI stands for Signal to Distortion Ratio Improvement.
        It is a measure of the improvement in signal-to-distortion ratio when using a denoising model.
        It is calculated as the difference between the signal-to-distortion ratio of the denoised signal and the signal-to-distortion ratio of the noisy signal.
        The higher the SI_SDRI, the better the denoising model.
    """
    # pred, target: shape (batch_size, 1, time_steps)
    pred, target = pred.squeeze(1), target.squeeze(1)
    target_energy = torch.sum(target**2, dim=-1, keepdim=True) + eps
    alpha = torch.sum(pred * target, dim=-1, keepdim=True) / target_energy
    target_scaled = alpha * target
    noise = pred - target_scaled
    si_sdr_val = 10 * torch.log10(
        (torch.sum(target_scaled**2, dim=-1) + eps) /
        (torch.sum(noise**2, dim=-1) + eps)
    )
    return -si_sdr_val.mean()  # negative for minimization

def spectral_l1(pred, target, n_fft=512, hop_length=128):
    """
    Spectral L1 loss is a measure of the difference between the magnitude spectra of the predicted and target signals.
        It is calculated as the L1 norm of the difference between the magnitude spectra of the predicted and target signals.
        The lower the spectral L1 loss, the better the denoising model.
    
    STFT stands for Short-Time Fourier Transform.
        This is the magnitude spectrum of the predicted and target signals.
    """
    pred_mag = torch.stft(pred.squeeze(1), 
                          n_fft=n_fft, 
                          hop_length=hop_length, 
                          return_complex=True).abs()
    
    target_mag = torch.stft(target.squeeze(1), 
                            n_fft=n_fft, 
                            hop_length=hop_length, 
                            return_complex=True).abs()
    
    return torch.mean(torch.abs(pred_mag - target_mag))

def l1_loss(pred, target):
    """L1 loss is a measure of the *absolute difference* between the predicted and target signals.
    It is calculated as the L1 norm of the difference between the predicted and target signals.
    The lower the L1 loss, the better the denoising model.

    Formula: torch.mean(torch.abs(pred - target))
    """
    return torch.nn.functional.l1_loss(pred, target)

def mse_loss(pred, target):
    """MSE loss is a measure of the *squared difference* between the predicted and target signals.
    It is calculated as the *mean squared error* between the predicted and target signals.
    The lower the MSE loss, the better the denoising model.

    Formula: torch.mean((pred - target) ** 2)
    """
    return torch.nn.functional.mse_loss(pred, target)

def gain_loss_rms_db(pred, target, eps=1e-8):
    """Gain loss is a measure of the difference in gain between the predicted and target signals.
    It is calculated as the *mean squared error* of the *difference in gain* between the predicted and target signals.
    The lower the gain loss, the better the denoising model.
    """
    # x: (batch_size, 1, time_steps)
    def rms(x, eps=1e-8):
        """Root Mean Square (RMS) is a measure of the *power* of a signal.
        It is calculated as the *square root of the mean of the squared values* of the signal.
        The lower the RMS, the better the denoising model.
        """
        return torch.sqrt(torch.mean(x**2, dim=(-1, -2)) + eps)  # (batch_size,)
    
    r_pred = rms(pred, eps)
    r_tgt  = rms(target, eps)
    delta_db = 20.0 * torch.log10((r_pred + eps) / (r_tgt + eps))
    return (delta_db**2).mean()
