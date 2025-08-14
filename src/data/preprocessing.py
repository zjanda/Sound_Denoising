import sys
import os
from tqdm import tqdm

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.dataset import AudioDataset
from src.data.manifest import create_manifest, load_manifest
from src.data.preprocessing_helpers import *

num_wave_files = 512

manifest = load_manifest('manifest_trainset_28spk_wav.json')
dataset = AudioDataset(manifest[:num_wave_files])

# Process and maintain the noisy/clean pair structure
processed_data = []
new_sample_rate = 16000

for i in tqdm(range(len(dataset))):
    # Get the original noisy/clean pair
    noisy_audio, clean_audio = dataset[i]  # This returns (noisy, clean)
    sample_rate = dataset.sample_rate
    # Process noisy audio
    noisy_processed = downsample(noisy_audio, orig_freq=sample_rate, new_freq=new_sample_rate)
    noisy_processed = normalize_amplitude(noisy_processed)
    noisy_mel_spec = convert_to_mel_spectrogram(noisy_processed, sample_rate=new_sample_rate)
    
    # Process clean audio  
    clean_processed = downsample(clean_audio, orig_freq=sample_rate, new_freq=new_sample_rate)
    clean_processed = normalize_amplitude(clean_processed)
    clean_mel_spec = convert_to_mel_spectrogram(clean_processed, sample_rate=new_sample_rate)
    
    # Maintain the pair structure
    processed_data.append((noisy_mel_spec, clean_mel_spec))

# Now processed_data is a list of tuples: [(noisy1, clean1), (noisy2, clean2), ...]
dataset = processed_data

print("Downsampling completed. Sample rate reduced and dataset shape:", getattr(dataset[0][0], 'shape', type(dataset[0][0])))
print("Amplitude normalization completed. Dataset stats - min:", getattr(dataset[0][0], 'min', lambda: None)(), "max:", getattr(dataset[0][0], 'max', lambda: None)())
print("Conversion to Mel Spectrogram completed. Dataset shape:", getattr(dataset[0][0], 'shape', type(dataset[0][0])))
print(f"Dataset now contains {len(dataset)} pairs of (noisy, clean) mel spectrograms")

# Save the processed dataset to a file
import torch

torch.save(dataset, 'src/data/processed_dataset.pt')

