import io
import os
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
import torchvision.transforms as standard_transforms
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from models import build_model


DEFAULT_WEIGHT_PATH = "./weights/SHTechA.pth"

app = FastAPI(title="Sistema de Contagem API", version="1.0.0")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_image(img_raw: Image.Image):
    width, height = img_raw.size
    new_width = max(128, width // 128 * 128)
    new_height = max(128, height // 128 * 128)

    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS

    resized = img_raw.resize((new_width, new_height), resample_filter)
    transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return resized, transform(resized)


@lru_cache(maxsize=1)
def load_model(weight_path: str):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Arquivo de pesos nao encontrado: {weight_path}")

    device = get_device()
    args = SimpleNamespace(backbone="vgg16_bn", row=2, line=2)
    model = build_model(args)
    model.to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device


def run_inference(model, device, img_raw: Image.Image, threshold: float):
    resized_img, img_tensor = preprocess_image(img_raw)
    samples = torch.Tensor(img_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
        outputs_points = outputs["pred_points"][0]

    mask = outputs_scores > threshold
    points = outputs_points[mask].detach().cpu().numpy().tolist()
    predict_cnt = int(mask.sum())
    return resized_img, points, predict_cnt


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    weight_path: str = Form(DEFAULT_WEIGHT_PATH),
) -> Dict[str, Any]:
    if not 0.1 <= threshold <= 0.9:
        raise HTTPException(status_code=400, detail="threshold deve estar entre 0.1 e 0.9")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Arquivo vazio")

    try:
        img_raw = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Arquivo enviado nao e uma imagem valida") from exc

    try:
        model, device = load_model(weight_path)
        resized_img, points, count = run_inference(model, device, img_raw, threshold)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao executar inferencia: {exc}") from exc

    return {
        "count": count,
        "points": points,
        "threshold": threshold,
        "device": str(device),
        "input_size": {"width": img_raw.width, "height": img_raw.height},
        "processed_size": {"width": resized_img.width, "height": resized_img.height},
        "filename": file.filename,
    }
