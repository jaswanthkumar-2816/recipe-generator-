import os
import sys
import urllib.request

expected_sizes = {
    'modelbest.ckpt': 415464764,
    'ingr_vocab.pkl': 30658,
    'instr_vocab.pkl': 464869
}

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
    
    success = True
    for filename, url in files_to_download.items():
        filepath = os.path.join(data_dir, filename)
        expected_size = expected_sizes.get(filename)
        
        # Verify existing file
        if os.path.exists(filepath):
            actual_size = os.path.getsize(filepath)
            if expected_size is None or actual_size == expected_size:
                print(f"{filename} already exists and is valid. Skipping download.")
                continue
            else:
                print(f"{filename} size mismatch (expected {expected_size}, got {actual_size}). Re-downloading...")
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Failed to remove corrupt file {filename}: {e}")
                    success = False
                    continue

        tmp_filepath = filepath + '.tmp'
        if os.path.exists(tmp_filepath):
            try:
                os.remove(tmp_filepath)
            except Exception:
                pass

        print(f"Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, tmp_filepath)
            # Verify downloaded file size
            if expected_size is not None:
                actual_size = os.path.getsize(tmp_filepath)
                if actual_size != expected_size:
                    raise ValueError(f"Downloaded file size mismatch (expected {expected_size}, got {actual_size})")
            
            os.rename(tmp_filepath, filepath)
            print(f"Successfully downloaded and verified {filename}.")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            if os.path.exists(tmp_filepath):
                try:
                    os.remove(tmp_filepath)
                except Exception:
                    pass
            success = False

    if not success:
        print("One or more model downloads failed. Exiting with error.")
        sys.exit(1)

if __name__ == "__main__":
    download_models()

