import os
import urllib.request

def download_models():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'Foodimg2Ing', 'data')
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Files to download
    files_to_download = {
        'modelbest.ckpt': 'https://dl.fbaipublicfiles.com/inversecooking/modelbest.ckpt',
        'ingr_vocab.pkl': 'https://dl.fbaipublicfiles.com/inversecooking/ingr_vocab.pkl',
        'instr_vocab.pkl': 'https://dl.fbaipublicfiles.com/inversecooking/instr_vocab.pkl'
    }
    
    for filename, url in files_to_download.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename} from {url}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Successfully downloaded {filename}.")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists. Skipping download.")

if __name__ == "__main__":
    download_models()
