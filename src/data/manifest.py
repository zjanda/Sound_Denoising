import json
import os
import pprint
import sys
import torchaudio
from pathlib import Path
from tqdm import tqdm

def get_data_dir():
    workspace_name = 'Sound_Denoising.code-workspace'
    # Step back in the directory tree until you find the workspace file
    # print('Path:', p / workspace_name)
    while not Path(workspace_name).exists():
        os.chdir('..')
        print('checking for workspace file in', os.getcwd(), '...')

    print('Found workspace file in', os.getcwd())
    with open(workspace_name, 'r') as f:
        data = json.load(f)
        data_dir = data['folders'][1]['path'] + '/'
    print('Data directory:', data_dir)
    return data_dir
    

def create_manifest(manifest_name: str, 
                    data_dir: str, 
                    sub_dir_context: str, 
                    save: bool = True) -> list[dict]:
    """
    Create a manifest for the given subdirectory.
    Args:
        manifest_name: str, name of the manifest to create
        data_dir: str, path to the data directory
        sub_dir_context: str, contextual string related to the subdirectories
            to create manifest for -> e.g. 'trainset_28spk_wav' -> 'clean_trainset_28spk_wav' and 'noisy_trainset_28spk_wav'
        save: bool, whether to save the manifest to a file
        debug: bool, whether to print debug information
    Returns:
        manifest: list[dict[str, str]], list of dictionaries containing the noisy (input) and clean (target) paths
    """
    manifest = []
    if 'clean_' in sub_dir_context:
        # drop the 'clean_' prefix from the dir_context if included
        sub_dir_context = sub_dir_context.replace('clean_', '')
    if 'noisy_' in sub_dir_context:
        # drop the 'noisy_' prefix from the dir_context if included
        sub_dir_context = sub_dir_context.replace('noisy_', '')

    path = data_dir + "clean_" + sub_dir_context

    if not Path(path).exists():
        raise Exception(f'Path {path} does not exist')
    
    # create a list of dictionaries containing the clean and noisy paths
    for wav_path in tqdm(Path(path).glob('**/*.wav'), desc=f'Creating manifest for {sub_dir_context}'):
        create_path = lambda x: str(data_dir + x + '_' + sub_dir_context + '/' + wav_path.name)
        manifest.append({
            'clean_path': create_path('clean'),
            'noisy_path': create_path('noisy'),
        })

    if save:
        print(f'Saving manifest to {manifest_name}')
        with open(manifest_name, 'w') as f:
            json.dump(manifest, f)
    return manifest

def load_manifest(manifest_name: str) -> list[dict]:
    with open(manifest_name, 'r') as f:
        return json.load(f)
        
if __name__ == '__main__':
    data_dir = get_data_dir()
    manifest_name = 'manifest_trainset_28spk_wav.json'
    manifest = create_manifest(manifest_name, data_dir, 'trainset_28spk_wav', save=True)
    print('len of manifest:', len(manifest))
    manifest_size = sys.getsizeof(manifest) / 1024 / 1024
    print(f'size of manifest: {manifest_size:.2f} MB')