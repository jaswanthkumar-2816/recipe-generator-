# import the necessary libraries

import matplotlib.pyplot as plt
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
    
    # Files to download if they don't exist
    files_to_download = {
        'modelbest.ckpt': 'https://dl.fbaipublicfiles.com/inversecooking/modelbest.ckpt',
        'ingr_vocab.pkl': 'https://dl.fbaipublicfiles.com/inversecooking/ingr_vocab.pkl',
        'instr_vocab.pkl': 'https://dl.fbaipublicfiles.com/inversecooking/instr_vocab.pkl'
    }
    
    for filename, url in files_to_download.items():
        filepath = os.path.join(_DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {filename}.")

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

    num_valid = 1
    title=[]
    ingredients=[]
    recipe=[]
    for i in range(num_valid):
        with torch.no_grad():
            outputs = model.sample(image_tensor, greedy=True,
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
            recipe.append("Reason: "+valid['reason'])

    return title,ingredients,recipe
