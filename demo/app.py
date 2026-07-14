import torch
import gradio as gr
import numpy as np
from PIL import Image
import torchvision.transforms as T
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from models.siamese_unet import SiameseUNet
from serving.monitoring import InferenceLogger, check_drift, image_stats, load_reference

# --- Monitoring setup ---
logger = InferenceLogger(os.path.join(os.path.dirname(__file__), 'inference_log.jsonl'))
REFERENCE = load_reference(os.path.join(os.path.dirname(__file__), '..', 'src', 'serving', 'reference_stats.json'))
import time as _time

# ImageNet normalization required by the ResNet34 encoder
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

TILE = 256          # patch size seen by the model during training (LEVIR-CD, ~0.5 m/px)
MAX_DIM = 2048       # caps the processed size to keep CPU inference fast

# Load the model
device    = torch.device('cpu')
model     = SiameseUNet(in_channels=3, pretrained=False).to(device)
ckpt_path = os.path.join(os.path.dirname(__file__), 'best_model.pt')
ckpt      = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Model loaded, training Val F1: {ckpt['val_f1']:.4f}")


def preprocess_tile(arr: np.ndarray) -> torch.Tensor:
    # arr: (256, 256, 3) uint8. No division by 255: the model was
    # trained on tensors normalized directly from the [0, 255] scale
    # (torchgeo LEVIRCDPlus loads images as raw floats, see dataset.py).
    t = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)  # (3, 256, 256)
    t = normalize(t)
    return t.unsqueeze(0)


def pad_to_tile_grid(img: Image.Image) -> tuple[Image.Image, int, int]:
    # Pad the image to a multiple of TILE (black border) for even tile splitting.
    w, h = img.size
    new_w = max(TILE, -(-w // TILE) * TILE)
    new_h = max(TILE, -(-h // TILE) * TILE)
    if (new_w, new_h) == (w, h):
        return img, w, h
    canvas = Image.new('RGB', (new_w, new_h))
    canvas.paste(img, (0, 0))
    return canvas, w, h


def predict(img_t1: Image.Image, img_t2: Image.Image):
    t_start = _time.perf_counter()
    img_t1 = img_t1.convert('RGB')
    img_t2 = img_t2.convert('RGB')

    # Cap the processed resolution, keeping the aspect ratio, to stay fast on CPU.
    w, h = img_t1.size
    scale = min(1.0, MAX_DIM / max(w, h))
    if scale < 1.0:
        img_t1 = img_t1.resize((int(w * scale), int(h * scale)))

    # Both images must be pixel-aligned to be compared tile by tile.
    img_t2 = img_t2.resize(img_t1.size)

    padded_t1, orig_w, orig_h = pad_to_tile_grid(img_t1)
    padded_t2, _, _           = pad_to_tile_grid(img_t2)
    W, H = padded_t1.size

    arr_t1 = np.array(padded_t1)
    arr_t2 = np.array(padded_t2)
    full_mask = np.zeros((H, W), dtype=np.float32)

    # Split into native 256x256 tiles (resolution seen during training) instead of
    # resizing the whole image, which would distort the real scale of buildings/roads.
    with torch.no_grad():
        for y in range(0, H, TILE):
            for x in range(0, W, TILE):
                tile1 = preprocess_tile(arr_t1[y:y+TILE, x:x+TILE]).to(device)
                tile2 = preprocess_tile(arr_t2[y:y+TILE, x:x+TILE]).to(device)
                logits = model(tile1, tile2)
                mask   = (torch.sigmoid(logits) > 0.5).float()
                full_mask[y:y+TILE, x:x+TILE] = mask[0, 0].cpu().numpy()

    # Crop back to the original size (removes the padding)
    full_mask = full_mask[:orig_h, :orig_w]
    t2_display = arr_t2[:orig_h, :orig_w]

    # Overlay: T2 with changed areas in red
    overlay = t2_display.copy()
    overlay[full_mask == 1] = [220, 30, 30]

    # --- Monitoring: log the inference and check input drift ---
    latency = _time.perf_counter() - t_start
    stats_t1 = image_stats(arr_t1[:orig_h, :orig_w])
    stats_t2 = image_stats(t2_display)
    change_ratio = float(full_mask.mean())

    warning = ""
    if REFERENCE is not None:
        drift = check_drift(stats_t1, REFERENCE)
        drift_t2 = check_drift(stats_t2, REFERENCE)
        if drift_t2.drifted:
            drift.drifted = True
            drift.reasons += [f"T2 {r}" for r in drift_t2.reasons]
        logger.log(latency, stats_t1, stats_t2, change_ratio, drift)
        if drift.drifted:
            warning = (
                "⚠️ **Input drift detected.** These images differ statistically from the "
                "LEVIR-CD+ training distribution (satellite RGB, ~0.5 m/px). "
                "The prediction may be unreliable.\n\n"
                + "\n".join(f"- {r}" for r in drift.reasons)
            )

    return (
        Image.fromarray(arr_t1[:orig_h, :orig_w]),
        Image.fromarray(t2_display),
        Image.fromarray(overlay),
        warning
    )


# Gradio interface
with gr.Blocks(title="Urban Change Detection") as demo:
    gr.Markdown("""
    # 🛰️ Urban Change Detection
    Urban change detection from satellite imagery.
    Upload two images of the same area at two different dates.
    The model detects new constructions and demolitions.

    **Architecture:** Siamese U-Net + pretrained ResNet34 · **Dataset:** LEVIR-CD+ · **Test F1:** 0.536
    """)

    with gr.Row():
        img_t1 = gr.Image(type='pil', label="Image T1 (Before)")
        img_t2 = gr.Image(type='pil', label="Image T2 (After)")

    btn = gr.Button("Detect changes", variant="primary")

    with gr.Row():
        out_t1      = gr.Image(label="T1 (before)")
        out_t2      = gr.Image(label="T2 (after)")
        out_overlay = gr.Image(label="Detected changes (red)")

    drift_warning = gr.Markdown()

    btn.click(
        fn      = predict,
        inputs  = [img_t1, img_t2],
        outputs = [out_t1, out_t2, out_overlay, drift_warning]
    )

    gr.Markdown("""
    ---
    **Jérémy Maille** · [GitHub](https://github.com/JeremyMaille/urban-change-detection)
    """)

demo.launch(server_name="0.0.0.0", server_port=7860)