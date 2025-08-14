import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, List, Callable, Union
import matplotlib.pyplot as plt

class ModelWrapper:
    def __init__(self, model, lr, weight_decay=None, step_size=None, gamma=None, optimizer=None, scheduler=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) if optimizer is None else optimizer
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma) if scheduler is None else scheduler
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.device = device
        self.parameters = model.parameters
        self.stopped_early = False
        self.name = str(model)

    def __call__(self, x):
        return self.model(x)
    
    def __str__(self):
        return str(self.model)

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv1d(cin, cout, 3, padding=1), nn.ReLU(),
        nn.GroupNorm(8, cout, affine=True),
        nn.Dropout(0.2),
    )

class AudioUNet_v1(nn.Module):
    def __init__(
        self,
        transforms: Optional[Callable] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        loss_fns: Optional[List[Callable]] = None,
        weights: Optional[List[float]] = None
    ) -> None:
        super().__init__()
        super().__init__()

        self.transforms = transforms
        self.device = device
        self.norm_layer = lambda x: nn.GroupNorm(8, x, affine=True)
        dropout = False
        self.dropout = nn.Dropout(0.2) if dropout else nn.Dropout(0.0)
        activation = True
        self.activation = nn.ReLU() if activation else nn.Identity()
        self.loss_fns = loss_fns
        self.weights = weights

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(), self.norm_layer(64), self.dropout,
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), self.norm_layer(128), self.dropout,
            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(), self.norm_layer(256), self.dropout,
            nn.Conv1d(256, 512, 3, padding=1), nn.ReLU(), self.norm_layer(512), self.dropout
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(512, 256, 3, padding=1), nn.ReLU(), self.norm_layer(256), self.dropout,
            nn.Conv1d(256, 128, 3, padding=1), nn.ReLU(), self.norm_layer(128), self.dropout,
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(), self.norm_layer(64), self.dropout,
            nn.Conv1d(64, 1, 3, padding=1), nn.Tanh()
        )
        self.to(self.device)
    
    def forward(self, X):
        if self.transforms is not None:
            X = self.transforms(X)
        
        # Encoder
        for layer in self.encoder:
            X = layer(X)
        
        # Decoder with skip connections
        for layer in self.decoder:
            X = layer(X)
        
        return X
    
    def loss_fn(self, pred, target):
        
        assert len(self.loss_fns) == len(self.weights), 'Number of loss functions and weights must match'
        total = 0.0
        for w, fn in zip(self.weights, self.loss_fns):
            total = total + w * fn(pred, target)
        weight_sum = sum(self.weights)
        return total / weight_sum
        
    def __str__(self):
        return f'AudioUNet_v1()'
    
class AudioUNet_v2(nn.Module):
    def __init__(self,
                 transforms=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 loss_fns=None,
                 weights=None):
        super().__init__()

        self.transforms = transforms
        self.device = device
        self.loss_fns = loss_fns
        self.weights = weights
        # self.norm_layer = lambda x: nn.GroupNorm(8, x, affine=True)

        # Encoder
        self.encoder = nn.Sequential(
            conv_block(1, 64),
            nn.Conv1d(64, 64, 4, stride=2, padding=1),
            conv_block(64, 128),
            nn.Conv1d(128, 128, 4, stride=2, padding=1),
            conv_block(128, 256),
            nn.Conv1d(256, 256, 4, stride=2, padding=1),
            conv_block(256, 512),
        )
        self.bottleneck = conv_block(512, 512)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(512, 512, 1),
            conv_block(512, 256),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(256, 256, 1),
            conv_block(256, 128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(128, 128, 1),
            conv_block(128, 64),
        )

        # Predict noise (residual); no Tanh
        self.noise_head = nn.Conv1d(64, 1, 1)
        nn.init.zeros_(self.noise_head.weight)
        nn.init.zeros_(self.noise_head.bias)
        self.res_scale = nn.Parameter(torch.tensor(1.0))

        self.to(self.device)
        
    def forward(self, x):
        if self.transforms is not None:
            x = self.transforms(x)
        
        enc_noise = x
        # Encoder
        for layer in self.encoder:
            enc_noise = layer(enc_noise)
        dec_noise = self.bottleneck(enc_noise)
        for layer in self.decoder:
            dec_noise = layer(dec_noise)
            
        noise = self.noise_head(dec_noise)
        denoised_x = x - self.res_scale * noise   # residual
        
        return denoised_x

    def loss_fn(self, pred, target):
        assert len(self.loss_fns) == len(self.weights), 'Number of loss functions and weights must match'
        total = 0.0
        for w, fn in zip(self.weights, self.loss_fns):
            total = total + w * fn(pred, target)
        weight_sum = sum(self.weights)
        return total / weight_sum

    def __str__(self):
        return f'AudioUNet_v2()'

