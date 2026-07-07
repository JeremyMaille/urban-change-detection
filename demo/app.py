import torch
import gradio as gr
import numpy as np
from PIL import Image
import torchvision.transforms as T
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from models.siamese_unet import SiameseUNet

# Normalisation ImageNet requise par l'encodeur ResNet34
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

TILE = 256          # taille de patch vue par le modèle à l'entraînement (LEVIR-CD, ~0.5 m/px)
MAX_DIM = 2048       # borne la taille traitée pour garder l'inférence CPU rapide

# Chargement du modèle
device    = torch.device('cpu')
model     = SiameseUNet(in_channels=3, pretrained=False).to(device)
ckpt_path = os.path.join(os.path.dirname(__file__), 'best_model.pt')
ckpt      = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Modèle chargé, Val F1 entraînement : {ckpt['val_f1']:.4f}")


def preprocess_tile(arr: np.ndarray) -> torch.Tensor:
    # arr : (256, 256, 3) uint8. Pas de division par 255 : le modèle a été
    # entraîné sur des tensors normalisés directement depuis l'échelle [0, 255]
    # (torchgeo LEVIRCDPlus charge les images en float brut, voir dataset.py).
    t = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)  # (3, 256, 256)
    t = normalize(t)
    return t.unsqueeze(0)


def pad_to_tile_grid(img: Image.Image) -> tuple[Image.Image, int, int]:
    # Complète l'image à un multiple de TILE (bord noir) pour un découpage en tuiles entières.
    w, h = img.size
    new_w = max(TILE, -(-w // TILE) * TILE)
    new_h = max(TILE, -(-h // TILE) * TILE)
    if (new_w, new_h) == (w, h):
        return img, w, h
    canvas = Image.new('RGB', (new_w, new_h))
    canvas.paste(img, (0, 0))
    return canvas, w, h


def predict(img_t1: Image.Image, img_t2: Image.Image):
    img_t1 = img_t1.convert('RGB')
    img_t2 = img_t2.convert('RGB')

    # Borne la résolution traitée, en conservant le ratio, pour rester rapide en CPU.
    w, h = img_t1.size
    scale = min(1.0, MAX_DIM / max(w, h))
    if scale < 1.0:
        img_t1 = img_t1.resize((int(w * scale), int(h * scale)))

    # Les deux images doivent être co-alignées pixel à pixel pour être comparées tuile par tuile.
    img_t2 = img_t2.resize(img_t1.size)

    padded_t1, orig_w, orig_h = pad_to_tile_grid(img_t1)
    padded_t2, _, _           = pad_to_tile_grid(img_t2)
    W, H = padded_t1.size

    arr_t1 = np.array(padded_t1)
    arr_t2 = np.array(padded_t2)
    full_mask = np.zeros((H, W), dtype=np.float32)

    # Découpe en tuiles natives 256×256 (résolution vue à l'entraînement) plutôt que
    # de resize l'image entière, ce qui écraserait l'échelle réelle des bâtiments/routes.
    with torch.no_grad():
        for y in range(0, H, TILE):
            for x in range(0, W, TILE):
                tile1 = preprocess_tile(arr_t1[y:y+TILE, x:x+TILE]).to(device)
                tile2 = preprocess_tile(arr_t2[y:y+TILE, x:x+TILE]).to(device)
                logits = model(tile1, tile2)
                mask   = (torch.sigmoid(logits) > 0.5).float()
                full_mask[y:y+TILE, x:x+TILE] = mask[0, 0].cpu().numpy()

    # Recadre au format original (retire le padding)
    full_mask = full_mask[:orig_h, :orig_w]
    t2_display = arr_t2[:orig_h, :orig_w]

    # Overlay : T2 avec les zones changées en rouge
    overlay = t2_display.copy()
    overlay[full_mask == 1] = [220, 30, 30]

    return (
        Image.fromarray(arr_t1[:orig_h, :orig_w]),
        Image.fromarray(t2_display),
        Image.fromarray(overlay)
    )


# Interface Gradio
with gr.Blocks(title="Urban Change Detection") as demo:
    gr.Markdown("""
    # 🛰️ Urban Change Detection
    Détection de changement urbain par imagerie satellite.
    Uploadez deux images de la même zone à deux dates différentes.
    Le modèle détecte les nouvelles constructions et démolitions.

    **Architecture :** Siamese U-Net + ResNet34 pré-entraîné · **Dataset :** LEVIR-CD+ · **Test F1 :** 0.536
    """)

    with gr.Row():
        img_t1 = gr.Image(type='pil', label="Image T1 — Avant")
        img_t2 = gr.Image(type='pil', label="Image T2 — Après")

    btn = gr.Button("Détecter les changements", variant="primary")

    with gr.Row():
        out_t1      = gr.Image(label="T1 (avant)")
        out_t2      = gr.Image(label="T2 (après)")
        out_overlay = gr.Image(label="Changements détectés (rouge)")

    btn.click(
        fn      = predict,
        inputs  = [img_t1, img_t2],
        outputs = [out_t1, out_t2, out_overlay]
    )

    gr.Markdown("""
    ---
    **Jérémy Maille** · [GitHub](https://github.com/JeremyMaille/urban-change-detection)
    """)

demo.launch()