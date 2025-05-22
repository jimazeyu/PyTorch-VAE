import os
import torch
import yaml
import argparse
from pathlib import Path
from models import *  # your model
from experiment import VAEXperiment
from dataset import VAEDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# --- Parse config --- #
parser = argparse.ArgumentParser(description='Evaluate VAE-generated images')
parser.add_argument('--config', '-c', dest="filename", metavar='FILE', default='configs/vae.yaml')
args = parser.parse_args()

with open(args.filename, 'r') as file:
    config = yaml.safe_load(file)

# --- Device & Params --- #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_SAMPLES = 100
LATENT_DIM = config['model_params']['latent_dim']

# --- Load Dataset --- #
data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()
real_loader = data.val_dataloader()

# --- Load Trained Model --- #
model = VAE(
    in_channels=config['model_params']['in_channels'],
    latent_dim=config['model_params']['latent_dim'],
    img_size=config['data_params']['patch_size']
).to(device)

ckpt_path = "./logs/VanillaVAE/version_0/checkpoints/last.ckpt"
state_dict = torch.load(ckpt_path, map_location=device)

# If checkpoint is a full PyTorch Lightning checkpoint:
if "state_dict" in state_dict:
    state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
model.load_state_dict(state_dict)
model.eval()

# --- Generate Fake Images --- #
generated_images = []
with torch.no_grad():
    for _ in range(NUM_SAMPLES // BATCH_SIZE):
        z = torch.randn(BATCH_SIZE, LATENT_DIM).to(device)
        samples = model.decode(z)
        samples = (samples + 1) / 2  # Assuming Tanh activation
        if samples.size(1) == 1:  # [B, 1, H, W] → [B, 3, H, W]
            samples = samples.repeat(1, 3, 1, 1)
        samples = F.interpolate(samples, size=(256, 256), mode='bilinear', align_corners=False)
        generated_images.append(samples.cpu())

fake_images = torch.cat(generated_images, dim=0)[:NUM_SAMPLES]

# --- Load Real Images --- #
real_images = []
for batch, _ in real_loader:
    batch = (batch + 1) / 2  # If originally normalized to [-1, 1]
    if batch.size(1) == 1:
        batch = batch.repeat(1, 3, 1, 1)
    batch = F.interpolate(batch, size=(256, 256), mode='bilinear', align_corners=False)
    real_images.append(batch)
    if len(real_images) * BATCH_SIZE >= NUM_SAMPLES:
        break

# --- Merge list to tensor ---
real_images = torch.cat(real_images, dim=0)[:NUM_SAMPLES]
fake_images = torch.cat(generated_images, dim=0)[:NUM_SAMPLES]

# --- Convert to uint8 for FID & IS ---
def convert_to_uint8(images):
    return (images.clamp(0, 1) * 255).to(torch.uint8)

real_images_uint8 = convert_to_uint8(real_images)
fake_images_uint8 = convert_to_uint8(fake_images)


fid = FrechetInceptionDistance(feature=2048).to(device)
iscore = InceptionScore().to(device)

fid.update(real_images_uint8.to(device), real=True)
fid.update(fake_images_uint8.to(device), real=False)

iscore.update(fake_images_uint8.to(device))

print("=" * 40)
print(f"FID Score: {fid.compute().item():.2f}")
is_mean, is_std = iscore.compute()
print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
print("=" * 40)
