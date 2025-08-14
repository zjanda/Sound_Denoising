import random
import time
import torch
from torch.utils.data import Dataset
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, manifest, transform=None, segment_ms=2000, sample_rate=16000, mono=True, max_samples=None):
        if manifest is None:
            raise ValueError("Manifest is required.")
        self.manifest = manifest[:max_samples] if max_samples else manifest
        self.transform = transform
        self.sample_rate = sample_rate
        self.segment_length = int(sample_rate * segment_ms / 1000.0)
        self.mono = mono    

    def __len__(self):
        return len(self.manifest)
    
    def _load_audio(self, file_path):
        if file_path.endswith('.wav'):
            waveform, sample_rate = torchaudio.load(file_path)
            if self.mono:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != self.sample_rate:
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)(waveform)
            return waveform
        else:
            return None

    def _random_or_center_crop(self, x, length):
        # x: [1, T]
        T = x.shape[-1]
        if T == length:
            return x
        if T > length:
            start = random.randint(0, T - length)
            return x[..., start:start+length]
        # pad if too short
        pad = length - T
        return torch.nn.functional.pad(x, (0, pad))

    def __getitem__(self, idx):
        item = self.manifest[idx]
        noisy_waveform = self._load_audio(item['noisy_path'])
        clean_waveform = self._load_audio(item['clean_path'])
        noisy_waveform = self._random_or_center_crop(noisy_waveform, self.segment_length)
        clean_waveform = self._random_or_center_crop(clean_waveform, self.segment_length)
        if self.transform:
            noisy_waveform = self.transform(noisy_waveform)
            clean_waveform = self.transform(clean_waveform)
        
        return noisy_waveform, clean_waveform
