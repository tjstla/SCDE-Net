import argparse
import os
from pathlib import Path

import cv2
import scipy.io as scio
import torch
import torch.nn.functional as F

from models import get_model


DATASET_ALIASES = {
    "irstd1k": "IRSTD-1k",
    "irstd-1k": "IRSTD-1k",
    "IRSTD-1k": "IRSTD-1k",
    "nudt": "NUDT-SIRST",
    "NUDT-SIRST": "NUDT-SIRST",
    "sirstaug": "sirst_aug",
    "sirst_aug": "sirst_aug",
    "SIRSTv1": "SIRSTv1",
    "sirstv1": "SIRSTv1",
}


DEFAULT_CHECKPOINTS = {
    "IRSTD-1k": "checkpoints/SCDE_IRSTD1K.pkl",
    "NUDT-SIRST": "checkpoints/SCDE_NUDT_SIRST.pkl",
    "sirst_aug": "checkpoints/SCDE_SIRSTAUG.pkl",
    "SIRSTv1": "checkpoints/SCDE_SIRSTv1.pkl",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run SCDE-Net inference.")
    parser.add_argument("--dataset", default="NUDT-SIRST", choices=sorted(DATASET_ALIASES))
    parser.add_argument("--checkpoint", default=None, help="Path to a model checkpoint.")
    parser.add_argument("--data-root", default="datasets", help="Root directory of datasets.")
    parser.add_argument("--out-dir", default="result", help="Directory for predicted masks and mat files.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=256)
    return parser.parse_args()


def normalize_dataset(name):
    return DATASET_ALIASES[name]


def resolve_file_list(data_root, dataset):
    data_root = Path(data_root)
    if dataset == "SIRSTv1":
        split_path = data_root / "SIRSTv1" / "Splits" / "test_v1.txt"
        image_root = data_root / "SIRSTv1" / "PNGImages"
        with split_path.open("r", encoding="utf-8") as f:
            file_list = [line.strip() + ".png" for line in f if line.strip()]
        return image_root, file_list

    image_root = data_root / dataset / "test" / "images"
    file_list = sorted([p.name for p in image_root.iterdir() if p.is_file()])
    return image_root, file_list


def load_checkpoint(model, checkpoint, device):
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model


def main():
    args = parse_args()
    dataset = normalize_dataset(args.dataset)
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[dataset]
    device = torch.device(args.device)

    model = get_model("scdenet")
    model = load_checkpoint(model, checkpoint, device)
    model.to(device)
    model.eval()

    image_root, file_list = resolve_file_list(args.data_root, dataset)
    image_out_dir = Path(args.out_dir) / dataset / "img"
    mat_out_dir = Path(args.out_dir) / dataset / "mat"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    mat_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, filename in enumerate(file_list, start=1):
        image_path = image_root / filename
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(image.reshape(1, 1, args.size, args.size) / 255.0).float().to(device)

        with torch.no_grad():
            _, target = model(tensor)
            target = F.sigmoid(target).detach().cpu().numpy().squeeze()
            target[target < 0] = 0

        stem = Path(filename).stem
        cv2.imwrite(str(image_out_dir / filename), target * 255)
        scio.savemat(str(mat_out_dir / f"{stem}.mat"), {"T": target})
        print(f"[{idx:04d}/{len(file_list):04d}] {filename}")

    print(f"Results saved to: {Path(args.out_dir) / dataset}")


if __name__ == "__main__":
    main()
