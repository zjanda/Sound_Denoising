import matplotlib.pyplot as plt
import torchaudio
import numpy as np

def visually_compare_audio_waveforms(front_waveform, back_waveform, sample_rate, title='Waveform Comparison', front_label='Front', back_label='Back', save_path=None):
    front_waveform = front_waveform.squeeze().numpy()
    duration_front = front_waveform.shape[-1] / sample_rate
    time_axis_front = np.linspace(0, duration_front, front_waveform.shape[-1])

    back_waveform = back_waveform.squeeze().numpy()
    duration_back = back_waveform.shape[-1] / sample_rate
    time_axis_back = np.linspace(0, duration_back, back_waveform.shape[-1])

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis_back, back_waveform, label=back_label)
    plt.plot(time_axis_front, front_waveform, label=front_label)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
        

def visually_compare_noisyandclean(data_dir, dataset, file_name, title='Waveform Comparison', front_label='Front', back_label='Back'):
    """
    Visualize the waveform of an audio file.

    Args:
        file_path (str): Path to the audio file (.wav).
        title (str, optional): Title for the plot.
    """
    clean_file = f'{data_dir}/clean_{dataset}/{file_name}'
    noisy_file = f'{data_dir}/noisy_{dataset}/{file_name}'
    waveform_front, front_sample_rate = torchaudio.load(clean_file)
    waveform_back, back_sample_rate = torchaudio.load(noisy_file)
    assert front_sample_rate == back_sample_rate, 'Sample rates must match'
    visually_compare_audio_waveforms(front_waveform=waveform_front, 
                                     back_waveform=waveform_back, 
                                     title=title, 
                                     front_label=front_label, 
                                     back_label=back_label,
                                     sample_rate=front_sample_rate)


if __name__ == '__main__':
    data_dir = 'D:/clean_noisy_sound_dataset/'
    dataset = 'trainset_28spk_wav'
    file_name = 'p226_001.wav'
    visually_compare_noisyandclean(data_dir, 
                                    dataset, 
                                    file_name, 
                                    title='Clean vs Noisy Waveform Comparison', 
                                    front_label='Clean', 
                                    back_label='Noisy')

