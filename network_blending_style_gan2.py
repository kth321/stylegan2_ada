# -*- coding: utf-8 -*-
"""network_blending-style gan2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EVEQJ3e35DhczllOZPxOdzsWv3SZbUO3

# StyleGAN2-ADA Network Blending

A user interface for experimenting with StyleGAN2-ADA network blending. If you're looking for StyleGAN3 blending, check out [this notebook](https://github.com/adamdavidcole/stylegan3-fun-blend/blob/main/blend.ipynb).

Select your source and destination models and play with various blend settings and sliders.

This notebook supports models of 256x256 pixels but could be extended to support larger outputs.

## Setup libraries and Google drive connection
"""

!nvidia-smi -L

# Connect Google Drive
# (NOTE: only run this if you want to save the results in GDrive after the runtime ends)
from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
import os
!pip install gdown --upgrade

if os.path.isdir("/content/drive/MyDrive/stylegan2-ada-pytorch-adam"):
#     %cd "/content/drive/MyDrive/stylegan2-ada-pytorch-adam"
elif os.path.isdir("/content/drive/"):
    #install script
#     %cd "/content/drive/MyDrive/"
    !git clone https://github.com/adamdavidcole/stylegan2-ada-pytorch-adam.git
#     %cd stylegan2-ada-pytorch-adam

    # !gdown --id 1-5xZkD8ajXw1DdopTkH_rAoCsD72LhKU -O /content/drive/MyDrive/colab-sg2-ada-pytorch/stylegan2-ada-pytorch/pretrained/wikiart.pkl
else:
    !git clone https://github.com/adamdavidcole/stylegan2-ada-pytorch-adam.git
#     %cd stylegan2-ada-pytorch-adam
    # !mkdir downloads
    # !mkdir datasets
    # !mkdir pretrained

!mkdir pretrained
!mkdir datasets
!mkdir input_images
!mkdir input_images/raw
!mkdir input_images/aligned

"""## Download Pretrained Models"""

import os
!pip install gdown --upgrade

# Download Various Pretrained Models
# Uncomment the group you'd like to download
# Mmake sure to use correct original and fine-tuned pair

if not os.path.isdir('pretrained'):
  !mkdir pretrained

# FFHQ 256
!wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl -O pretrained/ffhq_256.pkl

#  Butterflys (Trained form FFHQ_256)
!gdown --id 15NC-plFvfs59NLT0-t3SIucpJcEmvOmq -O pretrained/butterflys_000677.pkl

#  Pokémon (Trained form FFHQ_256)
!gdown --id 10sMVL02HibOs6fUdQSvUMQcs9h8mdsqz -O pretrained/pokemon_0503.pkl

# # Ukiyoe Face (Trained form FFHQ_256_SLIM)
!gdown --id 1BkRsnE0YygA2ufbfDOV4-fOgTMjSr94K -O pretrained/stylegan2-ffhq-slim.pkl
!gdown --id 1BjYGiOUKk8SC35a2e5QrJ1QtvaxJ0QD7 -O pretrained/ukiyoe-256-slim.pkl

# diseny
#!gdown --id 1z51gxECweWXqSYQxZJaHOJ4TtjUDGLxA -O pretrained/diseny.pkl

# FFHQ 256 version 2
!wget http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl -O pretrained/ffhq_256_2.pkl

"""## Setup Blend Functions and Utilities"""

#common functions
import pickle, torch, PIL, copy, cv2, math
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from google.colab import files
from io import BytesIO
from PIL import Image, ImageEnhance

from IPython.display import Image as DisplayImage, clear_output

# define device to use
device = torch.device('cuda')

def get_model(path):
  # with open(path, 'rb') as f:
  #   _G = pickle.load(f)['G_ema'].cuda()
  device = torch.device('cuda')
  with dnnlib.util.open_url(path) as fp:
      _G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)

  return _G

#tensor to PIL image
def t2i(t):
  return PIL.Image.fromarray((t*127.5+127).clamp(0,255)[0].permute(1,2,0).cpu().numpy().astype('uint8'))

#stack an array of PIL images horizontally
def add_imgs(images):
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = PIL.Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]
  return new_im


def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

# A simple color correction script to brighten overly dark images
def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)-1]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))-1]


        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

def normalize(inf, thresh):
    img = np.array(inf)
    out_img = simplest_cb(img, thresh)
    return PIL.Image.fromarray(out_img)

def get_w_from_path(w_path):
  projected_w_np = np.load(projected_w_path)[0]
  w = torch.tensor(projected_w_np).to(device).unsqueeze(0)
  return w

def synthesize_tensor_from_w(G, w):
  # print(w.shape)
  # print(w)
  return G.synthesis(w, noise_mode='const', force_fp32=True)

def synthesize_img_from_w(G, w):
  tensor = synthesize_tensor_from_w(G, w)
  return t2i(tensor)

def synthesize_tensor_from_w_path(G, w_path):
  w = get_w_from_path(w_path)
  return synthesize_tensor_from_w(G, w)

def synthesize_img_from_w_path(G, w_path):
  tensor = synthesize_tensor_from_w_path(G, w_path)
  return t2i(tensor)

def synthesize_img_from_w_path(G, w_path):
  tensor = synthesize_tensor_from_w_path(G, w_path)
  return t2i(tensor)

def synthesize_img_from_w_np(G, w_np):
  w = torch.tensor(w_np).to(device).unsqueeze(0)
  tensor = synthesize_img_from_w(G, w)
  return t2i(tensor)



class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

#@title Blend Functions

import os
import copy
import numpy as np
import torch
import pickle
import dnnlib
#import legacy2
import legacy

def extract_conv_names(model, model_res):
    model_names = list(name for name,weight in model.named_parameters())

    return model_names

def blend_models(low, high, model_res, resolution, level, blend_width=None, blend_mask=None, should_invert=False):

    resolutions =  [4*2**x for x in range(int(np.log2(resolution)-1))]

    low_names = extract_conv_names(low, model_res)
    high_names = extract_conv_names(high, model_res)

    assert all((x == y for x, y in zip(low_names, high_names)))

    #start with lower model and add weights above
    model_out = copy.deepcopy(low)
    params_src = high.named_parameters()
    dict_dest = model_out.state_dict()

    if blend_mask is None:
      for name, param in params_src:
          if should_invert:
            if any(f'synthesis.b{res}' in name for res in resolutions) and not ('mapping' in name):
              dict_dest[name].data.copy_(param.data)
          else:
            if not any(f'synthesis.b{res}' in name for res in resolutions) and not ('mapping' in name):
              dict_dest[name].data.copy_(param.data)

    else:
      for name, param in params_src:
        if not ('mapping' in name):
          # print(f"name: {name}")


          for idx, res in enumerate(resolutions):
            if f'synthesis.b{res}' in name:
              mask_val = blend_mask[idx]
              next_data = dict_dest[name].data * (1 - mask_val) + param.data * (mask_val)

              # print(mask_val)

              dict_dest[name].data.copy_(next_data)


    model_out_dict = model_out.state_dict()
    model_out_dict.update(dict_dest)
    model_out.load_state_dict(dict_dest)

    return model_out

"""## Select Models To Blend

**Select a source and destination model.**

Keep in mind that the destination model needs to be fine-tuned from the source model for the blend to work.

Feel free to paste in links to other pairs of models you'd like to expriment with.
"""

#@title {run: "auto"}
#@markdown Select a pretrained model for the source and destination or paste links to your own
#@markdown <br/>(Note: destination must be fine-tuned from source and both must be StyleGAN2 pkl format)
source_model = "FFHQ_256" #@param ["FFHQ_256","FFHQ_256_2","FFHQ_256_slim"] {allow-input: true}
destination_model = "Pokemon" #@param ["Butteflys", "Pokemon", "Ukiyoe_256_slim", "disney", "Cartoon"] {allow-input: true}


model_keys = {
    "FFHQ_256": "pretrained/ffhq_256.pkl",
    "FFHQ_256_2": "pretrained/ffhq_256_2.pkl",
    "Butteflys": "pretrained/butterflys_000677.pkl",
    "Pokemon": "pretrained/pokemon_0503.pkl",
    #"disney" : "/content/drive/MyDrive/stylegan2-ada-pytorch-adam/pretrained/diseny.pkl",
    "Cartoon" : "pretrained/ffhq-cartoon-blended-64.pkl",

    "FFHQ_256_slim": "/content/drive/MyDrive/stylegan2-ada-pytorch-adam/pretrained/stylegan2-ffhq-slim.pkl",
    "Ukiyoe_256_slim": "/content/drive/MyDrive/stylegan2-ada-pytorch-adam/pretrained/ukiyoe-256-slim.pkl"
}

lo_res_pkl = model_keys[source_model] if source_model in model_keys  else source_model
hi_res_pkl = model_keys[destination_model] if destination_model in model_keys else destination_model
model_res = 256
level = 0
blend_width=None

G_kwargs = dnnlib.EasyDict()

with dnnlib.util.open_url(lo_res_pkl) as f:
    # G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    lo = legacy.load_network_pkl(f, custom=False, **G_kwargs) # type: ignore
    lo_G, lo_D, lo_G_ema = lo['G'], lo['D'], lo['G_ema']


with dnnlib.util.open_url(hi_res_pkl) as f:
     #G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    hi = legacy.load_network_pkl(f, custom=False, **G_kwargs)['G_ema'] # type: ignore
    #hi_G, hi_D, hi_G_ema = hi['G'], lo['D'], lo['G_ema']

"""## Optional: Training
Optional: create a new source/fine-tuned set of models from the source model (only necessary if not using a pretrained models)
"""

### Training Settings
# (trained models output to the "results" folder)

# path to source model
resume_network = lo_res_pkl
# path to zip with prepped dataset
data_zip = "/content/drive/MyDrive/stylegan2-ada-pytorch-adam/input_images/raw/pokemon_256.zip"
# training config to use (for 256 images use "paper256")
cfg="paper256"
# Network snapshot frequenct (to see updates frequently use snap=1)
snap=1

!python train.py --outdir=results --data=$data_zip \
  --gpus=1 --cfg=$cfg --mirror=1 --snap=$snap --aug=noaug --metrics=none --resume=$resume_network

"""## Optional: Image Pojection

### Project Face
To project a custom face into the FFHQ latent space:


1.   Upload a square high res image to `input_images/raw`
2.   Run the `align_images.py` script below
3.   The aligned image will be output to `input_images/aligned`. Copy the path of the aligned image into the `uploaded_file_path` form option below
4.   Select a number of steps (5000 steps usually provides a high quality projectiob, but takes more time, 1000 is a good starting spot for testing)
5.   Run the `projector.py` cell below
6. Copy the path of the projected `.npy` file into the blending GUI forms below
"""

### Align Face Images

# Upload a file to input_images/raw
# Aligned image will be output to input_images/aligned
!rm -r -f input_images/raw/.ipynb_checkpoints
!python align_images.py

### Project Image Into Latent Space

import time
# ts stores the time in seconds
ts = int(time.time())

num_steps = 1003 #@param {type: "slider", min: 1, max: 10000, step: 1}
uploaded_file_path = "/content/drive/MyDrive/stylegan2-ada-pytorch-adam/input_images/aligned/myimage_01.png" #@param {type: "string"}

uploaded_file_name_with_ext = os.path.basename(uploaded_file_path)
uploaded_file_name = os.path.splitext(uploaded_file_name_with_ext)[0]


network_name_with_ext = os.path.basename(lo_res_pkl)
network_name = os.path.splitext(network_name_with_ext)[0]
projection_outdir = f"projections/{ts}__{network_name}__{uploaded_file_name}__{num_steps}"


!python projector.py --outdir=$projection_outdir --target=$uploaded_file_path --num-steps=$num_steps --save-video=false \
  --network=$lo_res_pkl

"""## Network Blending

### Network Blend Basic

Simple blend functions between the source and desination models. Select a seed value and blend mode.
"""

#@title Select Blend Layer {run: "auto"}
device = "cuda"

#@markdown **Optional: select source vector**
projected_w_path = "projections/1683463960__ffhq_256_2__myimage_01__1003/projected_w.npz" #@param {type: "string"}
use_projected_w = False #@param {type:"boolean"}
#@markdown ---
#@markdown **Select seed**
seed=7330 #@param {type: "slider", min: 0, max: 10000, step: 1}
truncation_psi = 0.8 #@param {type: "slider", min: 0, max: 1, step: 0.01}

#@markdown ---
#@markdown **Select blend layer options**
switch_layer = 4 #@param [4, 8, 16, 32, 64, 128]  {type:"raw"}
should_invert = False #@param {type: "boolean"}

# blend_width = 0 #@param {type: "slider", min: 0, max: 5, step: 0.01}

# switch_layer_inversion_map = {
#     4: 128,
#     8: 64,
#     16: 32,
#     32: 16,
#     64: 8,
#     128: 4
# }
# if invert_switch_layer:
#   switch_layer = switch_layer_inversion_map[switch_layer]

model_out = blend_models(lo_G_ema, hi, model_res, switch_layer, level, blend_width=blend_width, should_invert=should_invert)

G1 = lo_G_ema.to(device)
G2 = hi.to(device)
G_blend = model_out.to(device)


if use_projected_w:
  w_np = np.load(projected_w_path)['w']
  w = torch.tensor(w_np).to(device)
else:
  label = torch.zeros([1, G1.c_dim], device=device)
  z = torch.from_numpy(np.random.RandomState(seed).randn(1, G1.z_dim)).to(device)

  w = G1.mapping(z, None, truncation_psi=truncation_psi, truncation_cutoff=8)


g1_img = G1.synthesis(w, noise_mode='const', force_fp32=True)
g1_img = (g1_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
g1_imgfile = PIL.Image.fromarray(g1_img[0].cpu().numpy(), 'RGB')

# g1_imgfile.save(f'G1seed{seed:04d}.png')
g2_img = G2.synthesis(w, noise_mode='const', force_fp32=True)
g2_img = (g2_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
g2_imgfile = PIL.Image.fromarray(g2_img[0].cpu().numpy(), 'RGB')

g3_img = G_blend.synthesis(w, noise_mode='const', force_fp32=True)
g3_img = (g3_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
g3_imgfile = PIL.Image.fromarray(g3_img[0].cpu().numpy(), 'RGB')
display(add_imgs([g1_imgfile, g3_imgfile, g2_imgfile]))

"""### Fine Tune Blening

Fine tuned blending allows you to control the individual blend levels between the source and destination models. Higher sliders will control coarser, structual behavior while lower sliders control finer details like color and texture.
"""

#@title Select Blend Layer {run: "auto"}
device = "cuda"

#@markdown **Optional: Select source vector**
projected_w_path = "projections/1683438803__ffhq_256__myimage_01__1002/projected_w.npz" #@param {type: "string"}
use_projected_w = True #@param {type:"boolean"}
#@markdown ---
#@markdown **Select seed options**

seed=2271 #@param {type: "slider", min: 0, max: 10000, step: 1}
truncation_psi = 0.8 #@param {type: "slider", min: 0, max: 1, step: 0.01}
#@markdown ---
#@markdown **Select blend amount for each layer**

# switch_layer = 128 #@param [4, 8, 16, 32, 64, 128]  {type:"raw"}

blend_4 = 0.79 #@param {type: "slider", min: 0, max: 1, step: 0.01}
blend_8 = 0.79 #@param {type: "slider", min: 0, max: 1,  step: 0.01}
blend_16 = 0.79 #@param {type: "slider", min: 0, max: 1, step: 0.01}
blend_32 = 0.36 #@param {type: "slider", min: 0, max: 1, step: 0.01}
blend_64 = 0.7 #@param {type: "slider", min: 0, max: 1, step: 0.01}
blend_128 = 0.07 #@param {type: "slider", min: 0, max: 1, step: 0.01}
blend_256 = 1 #@param {type: "slider", min: 0, max: 1, step: 0.01}

blend_mask = [blend_4, blend_8, blend_16, blend_32, blend_64, blend_128, blend_256]
print(blend_mask)

model_out = blend_models(lo_G_ema, hi, model_res, model_res, level, blend_width=blend_width, blend_mask=blend_mask)

G1 = lo_G_ema.to(device)
G2 = hi.to(device)
G_blend = model_out.to(device)


if use_projected_w:
  w_np = np.load(projected_w_path)['w']
  w = torch.tensor(w_np).to(device)
else:
  label = torch.zeros([1, G1.c_dim], device=device)
  z = torch.from_numpy(np.random.RandomState(seed).randn(1, G1.z_dim)).to(device)

  print(f"Seed: {seed}")

  w = G1.mapping(z, None, truncation_psi=truncation_psi, truncation_cutoff=8)


g1_img = G1.synthesis(w, noise_mode='const', force_fp32=True)
g1_img = (g1_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
g1_imgfile = PIL.Image.fromarray(g1_img[0].cpu().numpy(), 'RGB')

# g1_imgfile.save(f'G1seed{seed:04d}.png')
g2_img = G2.synthesis(w, noise_mode='const', force_fp32=True)
g2_img = (g2_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
g2_imgfile = PIL.Image.fromarray(g2_img[0].cpu().numpy(), 'RGB')

g3_img = G_blend.synthesis(w, noise_mode='const', force_fp32=True)
g3_img = (g3_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
g3_imgfile = PIL.Image.fromarray(g3_img[0].cpu().numpy(), 'RGB')
display(add_imgs([g1_imgfile, g3_imgfile, g2_imgfile]))

"""### Experimental: Overblending
Overblending is a technique where you go beyond the sensible values of 0 and 1 when blending between the source and destination. The results often collapse when the sliders go far beyond the [0-1] scale, but it's fun to play with!
"""

#@title Select Blend Layer {run: "auto"}
device = "cuda"

#@markdown **Select source vector**
projected_w_path = "/content/drive/MyDrive/stylegan2-ada-pytorch-adam/projections/1655560119_ffhq_256/projected_w.npz" #@param {type: "string"}
use_projected_w = False #@param {type:"boolean"}
seed=2498 #@param {type: "slider", min: 0, max: 10000, step: 1}

#@markdown ---

# switch_layer = 128 #@param [4, 8, 16, 32, 64, 128]  {type:"raw"}

blend_4 = 1.23 #@param {type: "slider", min: -3, max: 3, step: 0.01}
blend_8 = 1.01 #@param {type: "slider", min: -3, max: 3, step: 0.01}
blend_16 = 0.94 #@param {type: "slider", min: -3, max: 3, step: 0.01}
blend_32 = -0.14 #@param {type: "slider", min: -3, max: 3, step: 0.01}
blend_64 = -0.14 #@param {type: "slider", min: -3, max: 3, step: 0.01}
blend_128 = -0.2 #@param {type: "slider", min: -3, max: 3, step: 0.01}
blend_256 = -0.13 #@param {type: "slider", min: -3, max: 3, step: 0.01}

blend_mask = [blend_4, blend_8, blend_16, blend_32, blend_64, blend_128, blend_256]
model_out = blend_models(lo_G_ema, hi, model_res, model_res, level, blend_width=blend_width, blend_mask=blend_mask)

blend_mask_clipped = np.clip(blend_mask, 0, 1)
model_out_clipped = blend_models(lo_G_ema, hi, model_res, model_res, level, blend_width=blend_width, blend_mask=blend_mask_clipped)

G1 = lo_G_ema.to(device)
G2 = hi.to(device)
G_blend = model_out.to(device)
G_blend_clipped = model_out_clipped.to(device)


if use_projected_w:
  w_np = np.load(projected_w_path)['w']
  w = torch.tensor(w_np).to(device)

  print(f"W: {'/'.join(projected_w_path.split('/')[-2:])}")
else:
  label = torch.zeros([1, G1.c_dim], device=device)
  z = torch.from_numpy(np.random.RandomState(seed).randn(1, G1.z_dim)).to(device)
  print(f"Seed: {seed}")
  w = G1.mapping(z, None, truncation_psi=0.8, truncation_cutoff=8)

print(blend_mask)


g1_img = G1.synthesis(w, noise_mode='const', force_fp32=True)
g1_img = (g1_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
g1_imgfile = PIL.Image.fromarray(g1_img[0].cpu().numpy(), 'RGB')

# g1_imgfile.save(f'G1seed{seed:04d}.png')
g2_img = G2.synthesis(w, noise_mode='const', force_fp32=True)
g2_img = (g2_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
g2_imgfile = PIL.Image.fromarray(g2_img[0].cpu().numpy(), 'RGB')

g3_img = G_blend.synthesis(w, noise_mode='const', force_fp32=True)
g3_img = (g3_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
g3_imgfile = PIL.Image.fromarray(g3_img[0].cpu().numpy(), 'RGB')

g4_img = G_blend_clipped.synthesis(w, noise_mode='const', force_fp32=True)
g4_img = (g4_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
g4_imgfile = PIL.Image.fromarray(g4_img[0].cpu().numpy(), 'RGB')



display(add_imgs([g1_imgfile, g3_imgfile, g4_imgfile, g2_imgfile]))

"""### Refrences

This notebook was created by [Adam Cole](https://www.instagram.com/adamcole.studio/) with a specific focus on building a user interface around network blending for artists to experiment with.

### Sources
- This code lives in a [fork of StyleGAN2](https://github.com/dvschultz/stylegan2-ada-pytorch) by [@dvschultz](https://github.com/dvschultz) and we take advantage of the training, projection, blending and utility function in that repo.
- The idea to use a "blend mask" and many helper functions were taken fully from [@Sxela](https://github.com/PDillis) [stylegan3_blending](https://github.com/Sxela/stylegan3_blending) repo
- Much of this work and some of the models were taken from Justin Pinkney's blogpost on network blending.
"""