import torch
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips
from pathlib import Path

# ps: pixel value should be in [0, 1]!


import argparse
import imageio.v3 as iio

from einops import rearrange
from torchvision.transforms import v2
from tqdm.auto import tqdm
import os
import torchmetrics.image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Directory containing generations for the model."
             "There should be subdirectories `samples_mp4` (generations) and `targets_mp4` (ground truth)."
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=50,
        help="Number of examples to evaluate on. Will evaluate in lexicographically decreasing order."
    )
    parser.add_argument(
        "--num_cond_frames",
        type=int,
        required=True,
        help="Number of conditioning frames, will be excluded from calculations."
    )
    return parser.parse_args()




def mp4_to_torch(mp4_path):
    frames_np = iio.imread(mp4_path, plugin="pyav")
    frames_torch = rearrange(torch.from_numpy(frames_np), "b h w c -> b c h w")
    return v2.functional.to_dtype(frames_torch, scale=True)


import json


if __name__ == "__main__":
    device = torch.device("cuda")
    args = parse_args()

    img_dir = Path(args.img_dir)
    samples_dir = img_dir / "samples_mp4"
    targets_dir = img_dir / "targets_mp4"

    all_samples = os.listdir(samples_dir)
    assert len(all_samples) >= args.num_examples, "Insufficient examples"
    final_samples = sorted(all_samples)[-args.num_examples:]
    final_targets = [sample.replace("samples", "targets") for sample in final_samples]

    assert all((targets_dir / final_target).exists() for final_target in final_targets), "Missing ground truth."

    videos1 = torch.stack([mp4_to_torch(samples_dir / sample)[args.num_cond_frames:] for sample in final_samples])
    videos2 = torch.stack([mp4_to_torch(targets_dir / target)[args.num_cond_frames:] for target in final_targets])


    result = {}
    result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv')
    # result['fvd'] = calculate_fvd(videos1, videos2, device, method='videogpt')
    result['ssim'] = calculate_ssim(videos1, videos2)
    result['psnr'] = calculate_psnr(videos1, videos2)
    result['lpips'] = calculate_lpips(videos1, videos2, device)
    print(json.dumps(result, indent=4))
