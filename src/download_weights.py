"""Download the DINOv2 backbone into weights/ (Apache-2.0, facebook/dinov2-base on Hugging Face).
python src/download_weights.py [--model facebook/dinov2-base]
For DINOv3 (gated; accept the license on Hugging Face and `hf auth login` first):
python src/download_weights.py --model facebook/dinov3-vitb16-pretrain-lvd1689m
then set "train.backbone" in the project config to weights/<model folder>.
"""
import argparse, os
from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="facebook/dinov2-base")
a = ap.parse_args()
dst = os.path.join(ROOT, "weights", a.model.split("/")[-1])
snapshot_download(a.model, local_dir=dst, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"])
print("weights ->", dst)
