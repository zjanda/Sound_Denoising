import torch
import torchaudio

def convert_to_mel_spectrogram(audio, sample_rate=16000, n_fft=2048, hop_length=512, n_mels=128):
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )(audio)
    return mel_spec

def downsample(audio, orig_freq=48000, new_freq=16000):
    """
    Downsample an audio tensor from orig_freq (default 48kHz) to new_freq (default 16kHz).
    Args:
        audio (Tensor): Audio tensor of shape (..., time)
        orig_freq (int): Original sampling rate (default 48000)
        new_freq (int): Target sampling rate (default 16000)
    Returns:
        Tensor: Downsampled audio tensor
    """
    resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
    return resampler(audio)

def normalize_amplitude(audio):
    # Normalize audio by amplitude so that the maximum absolute value is 1.0
    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio, dtype=torch.float32)
    max_amp = audio.abs().max()
    if max_amp > 0:
        audio = audio / max_amp
    return audio

def undo_button(mel_spec):
    # Create inverse transform
    inverse_transform = torchaudio.transforms.InverseMelScale(
        n_stft=1024 // 2 + 1,
        n_mels=mel_spec.shape[-2],
        sample_rate=16000,
    )

    # Use Griffin-Lim to recover waveform from spectrogram
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024)

    # Invert mel → linear
    linear_spec = inverse_transform(mel_spec)

    # Invert linear spectrogram → waveform
    return griffin_lim(linear_spec)