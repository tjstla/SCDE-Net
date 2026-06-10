import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps

from models import get_model
from models.DRPCANet_baseline import DRPCANetBaseline


DATASET_ALIASES = {
    "irstd1k": "IRSTD-1k",
    "irstd-1k": "IRSTD-1k",
    "IRSTD-1k": "IRSTD-1k",
    "nudt": "NUDT-SIRST",
    "NUDT-SIRST": "NUDT-SIRST",
    "sirstv1": "SIRSTv1",
    "SIRSTv1": "SIRSTv1",
}


DEFAULT_CHECKPOINTS = {
    "IRSTD-1k": "checkpoints/SCDE_IRSTD1K.pkl",
    "NUDT-SIRST": "checkpoints/SCDE_NUDT_SIRST.pkl",
    "SIRSTv1": "checkpoints/SCDE_SIRSTv1.pkl",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize sparse target response maps T^(1)..T^(K)."
    )
    parser.add_argument("--dataset", default="IRSTD-1k", choices=sorted(DATASET_ALIASES))
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--model-variant",
        default="current",
        choices=["current", "baseline"],
        help=(
            "current: use models/DRPCANet.py, which currently matches your modified model; "
            "baseline: use the original DRPCA-Net structure for baseline checkpoints"
        ),
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="feature_vis/stage_target_heatmap")
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--norm",
        default="prob",
        choices=["prob", "per-image", "shared-percentile"],
        help=(
            "prob: use sigmoid probability scale [0,1]; "
            "per-image: normalize each stage separately; "
            "shared-percentile: shared percentile scale across stages"
        ),
    )
    return parser.parse_args()


def normalize_dataset(name):
    return DATASET_ALIASES[name]


def resolve_image_path(root, dataset, image_name):
    stem = Path(image_name).stem
    suffix = Path(image_name).suffix or ".png"
    filename = stem + suffix
    if dataset == "SIRSTv1":
        candidates = [
            root / "datasets" / dataset / "PNGImages" / filename,
            root / "datasets" / dataset / "SIRST" / "images" / filename,
        ]
    else:
        candidates = [root / "datasets" / dataset / "test" / "images" / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find image {filename} in {dataset}.")


def resolve_mask_path(root, dataset, image_name):
    stem = Path(image_name).stem
    suffix = Path(image_name).suffix or ".png"
    filename = stem + suffix
    if dataset == "SIRSTv1":
        candidates = [
            root / "datasets" / dataset / "SIRST" / "BinaryMask" / f"{stem}_pixels0.png",
            root / "datasets" / dataset / "SIRST" / "masks" / filename,
            root / "datasets" / dataset / "masks" / filename,
        ]
    else:
        candidates = [root / "datasets" / dataset / "test" / "masks" / filename]
    for path in candidates:
        if path.exists():
            return path
    return None


def read_gray(path, size, interpolation):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.resize(image, (size, size), interpolation=interpolation)


def build_model(model_variant):
    if model_variant == "baseline":
        return DRPCANetBaseline(stage_num=6)
    return get_model("Drpcanet")


def load_model(root, checkpoint, device, model_variant):
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(model_variant)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, checkpoint_path


def to_numpy_2d(tensor):
    arr = tensor.detach().float().cpu()
    if arr.dim() == 4:
        arr = arr[0]
    if arr.dim() == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr.abs().mean(dim=0)
    return arr.numpy()


def normalize_minmax(arr, vmin=None, vmax=None):
    if vmin is None:
        vmin = float(arr.min())
    if vmax is None:
        vmax = float(arr.max())
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)


def heatmap(arr, cmap_name):
    rgb = (colormaps[cmap_name](arr)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def save_rgb(path, image_rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def collect_target_responses(model, inp):
    responses = []
    D = inp
    T = torch.zeros_like(D)
    for idx, stage in enumerate(model.decos, start=1):
        B = stage.lowrank(D, T)
        T = stage.sparse(D, B, T)
        responses.append({"stage": idx, "target": torch.sigmoid(T)})
        D = stage.merge(B, T)
    return responses


def render_overview(context_panels, stage_panels, save_path, title):
    panels = context_panels + stage_panels
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 3.1), dpi=240)
    for ax, item in zip(axes, panels):
        ax.imshow(item["image"])
        ax.set_title(item["title"], fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=10, y=0.99)
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.91))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    dataset = normalize_dataset(args.dataset)
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[dataset]
    device = torch.device(args.device)

    image_path = resolve_image_path(root, dataset, args.image)
    mask_path = resolve_mask_path(root, dataset, args.image)
    image = read_gray(image_path, args.size, cv2.INTER_LINEAR)
    mask = read_gray(mask_path, args.size, cv2.INTER_NEAREST) if mask_path else None
    if mask is not None and mask.max() > 0:
        mask = (mask / mask.max() * 255).astype(np.uint8)

    model, checkpoint_path = load_model(root, checkpoint, device, args.model_variant)
    inp = torch.from_numpy(image.reshape(1, 1, args.size, args.size) / 255.0).float().to(device)
    with torch.no_grad():
        responses = collect_target_responses(model, inp)

    stem = Path(args.image).stem
    out_dir = root / args.out_dir / dataset / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    target_arrays = [to_numpy_2d(item["target"]) for item in responses]
    if args.norm == "shared-percentile":
        values = np.concatenate([arr.reshape(-1) for arr in target_arrays])
        shared_vmin, shared_vmax = np.percentile(values, [0.5, 99.8])
    else:
        shared_vmin, shared_vmax = 0.0, 1.0

    origin_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    gt_rgb = cv2.cvtColor(mask if mask is not None else np.zeros_like(image), cv2.COLOR_GRAY2RGB)
    save_rgb(out_dir / "origin.png", origin_rgb)
    save_rgb(out_dir / "gt.png", gt_rgb)

    stage_panels = []
    for idx, arr in enumerate(target_arrays, start=1):
        if args.norm == "per-image":
            norm_arr = normalize_minmax(arr)
        else:
            norm_arr = normalize_minmax(arr, shared_vmin, shared_vmax)
        stage_rgb = heatmap(norm_arr, args.cmap)
        save_rgb(out_dir / f"T_stage{idx}.png", stage_rgb)
        stage_panels.append({"title": f"$T^{{({idx})}}$", "image": stage_rgb})

    render_overview(
        [{"title": "IRI", "image": origin_rgb}, {"title": "GT", "image": gt_rgb}],
        stage_panels,
        out_dir / "target_response_overview.png",
        f"Stage-wise sparse target response maps on {dataset} / {Path(args.image).name}",
    )

    print("Stage target response visualization finished.")
    print(f"Dataset      : {dataset}")
    print(f"Image        : {image_path}")
    print(f"Mask         : {mask_path if mask_path else 'not found'}")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"Model variant: {args.model_variant}")
    print(f"Normalization: {args.norm}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
