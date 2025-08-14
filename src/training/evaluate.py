import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import soundfile as sf


# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.model_classes import AudioUNet_v1
from src.training.loss_functions import si_sdr_loss, gain_loss_rms_db

class MelSpectrogramDataset(Dataset):
    """Dataset that converts 2D mel spectrograms to 1D format for Conv1D model."""
    
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        noisy_mel, clean_mel = self.data[idx]
        
        # Convert 2D mel spectrograms to 1D for Conv1D model
        # Input: [1, 128, 63] -> Output: [1, 128*63] = [1, 8064]
        noisy_1d = noisy_mel.reshape(1, -1)  # Flatten frequency and time dimensions
        clean_1d = clean_mel.reshape(1, -1)  # Flatten frequency and time dimensions
        
        return noisy_1d, clean_1d

def mel_collate_fn(batch):
    """Collate function that properly batches the 1D mel spectrograms."""
    noisy_list, clean_list = zip(*batch)
    
    # Stack into batches: [B, 1, 8064]
    noisy_batch = torch.stack(noisy_list, dim=0)
    clean_batch = torch.stack(clean_list, dim=0)
    
    return noisy_batch, clean_batch

class ModelEvaluator:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def evaluate_model(self, test_loader, loss_functions, loss_weights):
        """Evaluate model on test data."""
        total_losses = {name: 0.0 for name in loss_functions.keys()}
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (noisy_batch, clean_batch) in enumerate(tqdm(test_loader, desc="Evaluating")):
                noisy_batch = noisy_batch.to(self.device)
                clean_batch = clean_batch.to(self.device)
                
                # Forward pass (model expects [B, 1, length] format)
                denoised_batch = self.model(noisy_batch)
                
                # Calculate losses
                batch_losses = {}
                for name, loss_fn in loss_functions.items():
                    loss = loss_fn(denoised_batch, clean_batch)
                    batch_losses[name] = loss.item()
                    total_losses[name] += loss.item()
                
                total_samples += noisy_batch.size(0)
                
                # Print first batch details
                if batch_idx == 0:
                    print(f"\nFirst batch shapes:")
                    print(f"  Noisy input: {noisy_batch.shape}")
                    print(f"  Clean target: {clean_batch.shape}")
                    print(f"  Denoised output: {denoised_batch.shape}")
                    print(f"  First batch losses: {batch_losses}")
        
        # Calculate average losses
        avg_losses = {name: loss / total_samples for name, loss in total_losses.items()}
        return avg_losses
    
    def generate_samples(self, test_loader, num_samples=5):
        """Generate sample denoised outputs for visualization."""
        samples = []
        
        with torch.no_grad():
            for batch_idx, (noisy_batch, clean_batch) in enumerate(test_loader):
                if batch_idx >= num_samples:
                    break
                    
                noisy_batch = noisy_batch.to(self.device)
                clean_batch = clean_batch.to(self.device)
                
                # Forward pass
                denoised_batch = self.model(noisy_batch)
                
                samples.append({
                    'noisy': noisy_batch.cpu(),
                    'clean': clean_batch.cpu(),
                    'denoised': denoised_batch.cpu()
                })
        
        return samples
    
    def visualize_samples(self, samples, save_path='evaluation_samples.png'):
        """Visualize sample results as 1D signals."""
        fig, axes = plt.subplots(len(samples), 3, figsize=(15, 5*len(samples)))
        
        if len(samples) == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample in enumerate(samples):
            # Plot 1D mel spectrograms (flattened)
            axes[i, 0].plot(sample['noisy'][0, 0].numpy())
            axes[i, 0].set_title(f'Sample {i+1}: Noisy (1D)')
            axes[i, 0].set_ylabel('Amplitude')
            
            axes[i, 1].plot(sample['clean'][0, 0].numpy())
            axes[i, 1].set_title(f'Sample {i+1}: Clean (1D)')
            
            axes[i, 2].plot(sample['denoised'][0, 0].numpy())
            axes[i, 2].set_title(f'Sample {i+1}: Denoised (1D)')
            axes[i, 2].set_xlabel('Time Steps (8064)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main evaluation function."""
    print("Starting model evaluation...")
    
    # Load preprocessed data
    data_path = 'src/data/processed_dataset.pt'
    if not os.path.exists(data_path):
        print(f"Error: Processed data not found at {data_path}")
        print("Please run preprocessing first!")
        return
    
    print(f"Loading processed data from {data_path}")
    processed_data = torch.load(data_path)
    print(f"Loaded {len(processed_data)} data samples")
    
    # Check data structure
    if len(processed_data) > 0:
        first_item = processed_data[0]
        print(f"First item type: {type(first_item)}")
        if isinstance(first_item, tuple):
            print(f"First item shapes: {[item.shape for item in first_item]}")
            print(f"Expected: (noisy_mel, clean_mel) where each is [1, 128, 63]")
    
    # Create dataset that converts 2D to 1D
    dataset = MelSpectrogramDataset(processed_data)
    
    # Split into train/val/test
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create dataloaders
    test_loader = DataLoader(
        test_dataset, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=mel_collate_fn
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AudioUNet_v1()
    print(f"Model initialized: {model}")
    print(f"Model expects input shape: [batch_size, 1, length]")
    
    # Define loss functions
    loss_functions = {
        'SI-SDR': si_sdr_loss,
        'RMS_Gain': gain_loss_rms_db
    }
    loss_weights = [0.7, 0.3]
    
    # Evaluate model
    evaluator = ModelEvaluator(model, device)
    
    print("\n" + "="*50)
    print("EVALUATING MODEL ON TEST SET")
    print("="*50)
    
    avg_losses = evaluator.evaluate_model(test_loader, loss_functions, loss_weights)
    
    print(f"\nTest Set Results:")
    for loss_name, loss_value in avg_losses.items():
        print(f"  {loss_name}: {loss_value:.6f}")
    
    # Generate and visualize samples
    print(f"\nGenerating sample visualizations...")
    samples = evaluator.generate_samples(test_loader, num_samples=3)
    evaluator.visualize_samples(samples, 'test_evaluation_samples.png')
    
    print(f"\nEvaluation complete! Check 'test_evaluation_samples.png' for visual results.")

    # Create output directory for audio samples
    output_dir = "test_evaluation_audio"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving audible samples to '{output_dir}'...")

    # Import torchaudio for mel spectrogram conversion
    import torchaudio
    import torchaudio.transforms as T

    # Create inverse mel spectrogram transform
    mel_scale = T.MelScale(
        sample_rate=16000,
        n_fft=1024,
        n_mels=128,
        f_min=0,
        f_max=8000
    )
    
    # Create inverse transform (approximate)
    inverse_mel = T.InverseMelScale(
        sample_rate=16000,
        n_fft=1024,
        n_mels=128,
        f_min=0,
        f_max=8000
    )

    # Each sample is a dictionary: {'noisy': tensor, 'clean': tensor, 'denoised': tensor}
    for idx, sample in enumerate(samples):
        # Extract tensors from the sample dictionary
        noisy = sample['noisy']
        clean = sample['clean'] 
        denoised = sample['denoised']
        
        # Convert 1D flattened mel spectrograms back to 2D [1, 128, 63]
        batch_size = noisy.shape[0]
        noisy_2d = noisy.reshape(batch_size, 1, 128, 63)
        clean_2d = clean.reshape(batch_size, 1, 128, 63)
        denoised_2d = denoised.reshape(batch_size, 1, 128, 63)
        
        # Take first sample from batch
        noisy_2d = noisy_2d[0]  # [1, 128, 63]
        clean_2d = clean_2d[0]   # [1, 128, 63]
        denoised_2d = denoised_2d[0]  # [1, 128, 63]
        
        try:
            # Convert mel spectrograms back to raw audio (approximate)
            # This is a simplified conversion - for better quality you'd need proper inverse mel
            noisy_audio = noisy_2d.mean(dim=0)  # Average across frequency bins
            clean_audio = clean_2d.mean(dim=0)   # Average across frequency bins
            denoised_audio = denoised_2d.mean(dim=0)  # Average across frequency bins
            
            # Convert to numpy and normalize
            def norm_audio(x):
                x = x.cpu().numpy().astype(np.float32)
                max_val = np.max(np.abs(x))
                return x / max_val if max_val > 0 else x

            noisy_np = norm_audio(noisy_audio)
            clean_np = norm_audio(clean_audio)
            denoised_np = norm_audio(denoised_audio)

            # Save as wav files (assume 16kHz sample rate)
            sf.write(os.path.join(output_dir, f"sample{idx+1}_noisy.wav"), noisy_np, 16000)
            sf.write(os.path.join(output_dir, f"sample{idx+1}_clean.wav"), clean_np, 16000)
            sf.write(os.path.join(output_dir, f"sample{idx+1}_denoised.wav"), denoised_np, 16000)
            
            print(f"Saved sample {idx+1} successfully")
            
        except Exception as e:
            print(f"Error processing sample {idx+1}: {e}")
            continue

    print(f"Saved {len(samples)} sets of (noisy, clean, denoised) audio samples to '{output_dir}'.")
    print("Note: These are simplified conversions from mel spectrograms to audio.")
    print("For better quality, implement proper inverse mel spectrogram conversion.")

if __name__ == "__main__":
    main()
