# import the necessary libraries

# import matplotlib.pyplot as plt # Removed to save memory
import torch
import torch.nn as nn
import numpy as np
import os
from Foodimg2Ing.args import get_parser
import pickle
from Foodimg2Ing.model import get_model
from torchvision import transforms
from Foodimg2Ing.utils.output_utils import prepare_output
from PIL import Image
import time
from Foodimg2Ing import app


# ---- Global caches (loaded once per Gunicorn worker) ----
_DATA_DIR = os.path.join(app.root_path, 'data')
_INGRS_VOCAB = None
_INSTR_VOCAB = None
_MODEL = None
_DEVICE = None
_TO_INPUT_TRANSF = None


def _get_assets():
    global _INGRS_VOCAB, _INSTR_VOCAB, _MODEL, _DEVICE, _TO_INPUT_TRANSF

    if _MODEL is not None:
        return _MODEL, _DEVICE, _INGRS_VOCAB, _INSTR_VOCAB, _TO_INPUT_TRANSF

    import urllib.request
    import torch
    
    # Set PyTorch to use 1 thread to save memory on Render's free tier
    torch.set_num_threads(1)
    
    # Ensure data directory exists
    os.makedirs(_DATA_DIR, exist_ok=True)
    
    expected_sizes = {
        'modelbest.ckpt': 415464764,
        'ingr_vocab.pkl': 30658,
        'instr_vocab.pkl': 464869
    }
    
    # Files to download if they don't exist
    files_to_download = {
        'modelbest.ckpt': 'https://dl.fbaipublicfiles.com/inversecooking/modelbest.ckpt',
        'ingr_vocab.pkl': 'https://dl.fbaipublicfiles.com/inversecooking/ingr_vocab.pkl',
        'instr_vocab.pkl': 'https://dl.fbaipublicfiles.com/inversecooking/instr_vocab.pkl'
    }
    
    for filename, url in files_to_download.items():
        filepath = os.path.join(_DATA_DIR, filename)
        expected_size = expected_sizes.get(filename)
        
        # Verify existing file size/integrity
        if os.path.exists(filepath):
            actual_size = os.path.getsize(filepath)
            if expected_size is None or actual_size == expected_size:
                continue
            else:
                print(f"{filename} size mismatch (expected {expected_size}, got {actual_size}). Re-downloading...")
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Failed to remove corrupt file {filename}: {e}")
                    continue

        tmp_filepath = filepath + '.tmp'
        if os.path.exists(tmp_filepath):
            try:
                os.remove(tmp_filepath)
            except Exception:
                pass

        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, tmp_filepath)
            # Verify size
            if expected_size is not None:
                actual_size = os.path.getsize(tmp_filepath)
                if actual_size != expected_size:
                    raise ValueError(f"Downloaded file size mismatch (expected {expected_size}, got {actual_size})")
            os.rename(tmp_filepath, filepath)
            print(f"Downloaded and verified {filename}.")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            if os.path.exists(tmp_filepath):
                try:
                    os.remove(tmp_filepath)
                except Exception:
                    pass
            raise e

    use_gpu = True
    _DEVICE = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'

    _INGRS_VOCAB = pickle.load(open(os.path.join(_DATA_DIR, 'ingr_vocab.pkl'), 'rb'))
    _INSTR_VOCAB = pickle.load(open(os.path.join(_DATA_DIR, 'instr_vocab.pkl'), 'rb'))

    ingr_vocab_size = len(_INGRS_VOCAB)
    instrs_vocab_size = len(_INSTR_VOCAB)

    import sys
    sys.argv = ['']
    args = get_parser()
    args.maxseqlen = 15
    args.ingrs_only = False

    _MODEL = get_model(args, ingr_vocab_size, instrs_vocab_size)

    # Load the pre-trained model parameters
    model_path = os.path.join(_DATA_DIR, 'modelbest.ckpt')
    state_dict = torch.load(model_path, map_location=map_loc)
    _MODEL.load_state_dict(state_dict)
    
    # Free up memory
    del state_dict
    import gc
    gc.collect()

    _MODEL.to(_DEVICE)
    _MODEL.eval()
    _MODEL.ingrs_only = False
    _MODEL.recipe_only = False

    # Precompute tensor transform
    _TO_INPUT_TRANSF = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return _MODEL, _DEVICE, _INGRS_VOCAB, _INSTR_VOCAB, _TO_INPUT_TRANSF


def output(uploadedfile):

    # Keep all the codes and pre-trained weights in data directory
    data_dir = _DATA_DIR

    model, device, ingrs_vocab, vocab, to_input_transf = _get_assets()

    uploaded_file = uploadedfile

    # Load image via PIL (RGB)
    img = Image.open(uploaded_file).convert('RGB')

    show_anyways = False #if True, it will show the recipe even if it's not valid
    transf_list = []
    transf_list.append(transforms.Resize(256))
    transf_list.append(transforms.CenterCrop(224))
    transform = transforms.Compose(transf_list)

    image_transf = transform(img)
    image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)

    num_valid = 2
    title=[]
    ingredients=[]
    recipe=[]
    for i in range(num_valid):
        with torch.no_grad():
            # For the first one use greedy, for the second use random sampling to get a different result
            greedy_sample = (i == 0)
            outputs = model.sample(image_tensor, greedy=greedy_sample,
                                temperature=1.0, beam=-1, true_ingrs=None)

        ingr_ids = outputs['ingr_ids'].cpu().numpy()
        recipe_ids = outputs['recipe_ids'].cpu().numpy()

        outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingrs_vocab, vocab)

        if valid['is_valid'] or show_anyways:
            title.append(outs['title'])
            ingredients.append(outs['ingrs'])
            recipe.append(outs['recipe'])
        else:
            title.append("Not a valid recipe!")
            ingredients.append([]) # Append empty list to avoid IndexError in template
            recipe.append("Reason: "+valid['reason'])

    return title,ingredients,recipe
