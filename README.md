# 🎵 Sound Denoising with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## �� Overview

Advanced audio denoising system using U-Net architecture with PyTorch. This project demonstrates state-of-the-art techniques for removing noise from audio recordings while preserving audio quality.

## ✨ Features

- **U-Net Architecture**: Custom AudioUNet implementation for audio processing
- **Multiple Loss Functions**: SI-SDR, RMS gain, and custom loss combinations
- **Data Augmentation**: Advanced audio preprocessing and augmentation
- **K-Fold Validation**: Robust model evaluation with cross-validation
- **Real-time Processing**: Live audio denoising capabilities

## 🏗️ Architecture

- **Encoder**: 4-layer convolutional network with increasing channels (64→512)
- **Decoder**: 4-layer deconvolutional network with skip connections
- **Normalization**: GroupNorm for stable training
- **Activation**: ReLU with dropout for regularization

## 📊 Results

- **Training Loss**: [Your metrics here]
- **Validation Loss**: [Your metrics here]
- **Audio Quality**: [PESQ/STOI scores if available]

## ��️ Installation

```bash
git clone https://github.com/yourusername/Sound_Denoising.git
cd Sound_Denoising
pip install -r requirements.txt
```

## 📖 Usage

### Basic Training
```python
from src.models.model_classes import AudioUNet_v1
from src.training.helpers import train_model

model = AudioUNet_v1()
train_model(model, train_loader, val_loader, epochs=100)
```

### Live Denoising
```python
from src.demonstrative.live_sound_visualizer import LiveDenoiser

denoiser = LiveDenoiser()
denoiser.start()
```

## �� Testing

```bash
pytest tests/
```

## �� Documentation

See [docs/](docs/) for detailed documentation.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## �� Acknowledgments

- VoiceBank-DEMAND dataset
- PyTorch community
- Audio processing research community