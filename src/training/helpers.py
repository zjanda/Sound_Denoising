import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

def get_dataloader(dataset, batch_size, shuffle, num_workers=None, device="cuda" if torch.cuda.is_available() else "cpu", debug=True):
    if num_workers is None:
        import os
        cpu_count = os.cpu_count() or 4 # default to 4 if os.cpu_count() is None
        # print('cpu_count =', cpu_count)
        num_workers = max(2, cpu_count // 2) # default to 2 if cpu_count is less than 4
    if debug:
        num_workers = 0 
    # print('num_workers =', num_workers)
    pin = (torch.cuda.is_available() and 'cuda' in str(device))
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        persistent_workers=(num_workers > 0),
                        prefetch_factor=2 if num_workers > 0 else None,
                        pin_memory=pin,
                        drop_last=True)
    return loader


def stft_mag(x, n_fft, hop):
    win = torch.hann_window(n_fft, device=x.device)
    X = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=hop, window=win,
                   return_complex=True, center=True)
    return X.abs()

def mr_stft_loss(pred, target):
    cfgs = [(256,64), (512,128), (1024,256)]
    loss = 0.0
    for n_fft, hop in cfgs:
        loss += F.l1_loss(stft_mag(pred, n_fft, hop), stft_mag(target, n_fft, hop))
    return loss / len(cfgs)

def rms(x, eps=1e-8):
    return torch.sqrt(torch.mean(x**2, dim=(-1,-2), keepdim=True) + eps)

def denoise_loss(pred, target):
    L_time = 0.5 * F.l1_loss(pred, target) + 0.5*F.mse_loss(pred, target)
    L_spec = mr_stft_loss(pred, target)
    L_gain = F.mse_loss(rms(pred), rms(target))
    return L_time + 0.5*L_spec + 0.01*L_gain

def train_model(model, train_loader, optimizer, loss_fn, plot_updater=None):
    if hasattr(model, 'loss_fn') and model.loss_fn is not None:
        loss_fn = model.loss_fn
    model.train()
    total_loss, samples = 0.0, 0

    start_loading_data = time.time()
    loading_data = True
    data_load_time = None
    pbar = tqdm(train_loader, desc='Training', total=len(train_loader))
    for X_batch, y_batch in pbar:
        if loading_data:
            data_load_time = time.time() - start_loading_data
            loading_data = False
        if data_load_time is not None:
            pbar.set_postfix_str(f"Data load time: {data_load_time:.2f}s")

        batch_size = X_batch.shape[0]
        X_batch, y_batch = X_batch.reshape(batch_size, 1, -1).to(model.device), y_batch.reshape(batch_size, 1, -1).to(model.device)
        optimizer.zero_grad()
        
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        batch_size = X_batch.size(0)
        samples += batch_size
        total_loss += loss.item() * batch_size
        avg_loss = total_loss / max(1,samples)

        # Update live display
        if plot_updater is not None:
            plot_updater(avg_loss)

    return avg_loss

def evaluate_model(model, val_loader, loss_fn=None, plot_updater=None):
    if loss_fn is None and hasattr(model, 'loss_fn') and model.loss_fn is not None:
        loss_fn = model.loss_fn
    model.eval()
    total_loss, samples = 0.0, 0
    
    loading_data = True
    data_load_time = None
    start_loading_data = time.time()
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating', total=len(val_loader))
        for X_batch, y_batch in pbar:
            
            if loading_data:
                data_load_time = time.time() - start_loading_data
                loading_data = False
            if data_load_time is not None:
                pbar.set_postfix_str(f"Data load time: {data_load_time:.2f}s")
                
            batch_size = X_batch.shape[0]
            X_batch, y_batch = X_batch.reshape(batch_size, 1, -1).to(model.device), y_batch.reshape(batch_size, 1, -1).to(model.device)
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            
            batch_size = X_batch.size(0)
            samples += batch_size
            total_loss += loss.item() * batch_size
            avg_loss = total_loss / max(1,samples)

            # Update live display
            if plot_updater is not None:
                plot_updater(avg_loss)
    return avg_loss
