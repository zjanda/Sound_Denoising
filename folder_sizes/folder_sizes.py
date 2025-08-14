import os
from tqdm import tqdm

def get_folder_sizes(directory):
    folder_sizes = {}
    print("Getting folder sizes...")
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(folder_path):
                print(f'Processing {dirpath}')
                for file in tqdm(filenames):
                    file_path = os.path.join(dirpath, file)
                    total_size += os.path.getsize(file_path)

            folder_sizes[folder_name] = total_size
    return folder_sizes

def display_folder_sizes(directory):
    folder_sizes = get_folder_sizes(directory)
    for folder, size in folder_sizes.items():
        print(f"Folder: {folder}, Size: {size / (1024 ** 2):.2f} MB")

if __name__ == "__main__":
    base_directory = 'D:/denoise_sound_files/'
    display_folder_sizes(base_directory)


# # Essential parameters
# tqdm(
#     iterable,           # The iterable to wrap
#     total=None,         # Total number of iterations
#     desc=None,          # Description string
#     leave=True,         # Whether to leave the progress bar after completion
#     file=None,          # Output file (default: sys.stderr)
#     ncols=None,         # Width of the progress bar
#     mininterval=0.1,    # Minimum progress display update interval
#     maxinterval=10.0,   # Maximum progress display update interval
#     miniters=None,      # Minimum iterations between display updates
#     ascii=None,         # Use ASCII characters instead of Unicode
#     unit='it',          # Unit of iteration
#     unit_scale=False,   # Scale the unit (e.g., 'K', 'M', 'G')
#     unit_divisor=1000,  # Divisor for unit scaling
#     dynamic_ncols=False, # Allow dynamic column width
#     smoothing=0.3,      # Exponential moving average smoothing
#     bar_format=None,    # Custom bar format string
#     initial=0,          # Initial counter value
#     position=None,      # Position of the progress bar
#     postfix=None,       # Additional info to display
#     gui=False,          # Use GUI progress bar
#     **kwargs
# )