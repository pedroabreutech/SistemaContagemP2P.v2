import io
import os
import tempfile
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from starlette.background import BackgroundTask

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


def draw_points_on_image(img: Image.Image, points: List[List[float]]) -> np.ndarray:
    img_to_draw = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), 2, (57, 255, 20), -1)
    return img_to_draw


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


@app.post("/predict-video")
async def predict_video(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    weight_path: str = Form(DEFAULT_WEIGHT_PATH),
    frame_step: int = Form(1),
    line_orientation: str = Form("horizontal"),
    line_ratio: float = Form(0.6),
) -> FileResponse:
    if not 0.1 <= threshold <= 0.9:
        raise HTTPException(status_code=400, detail="threshold deve estar entre 0.1 e 0.9")
    if frame_step < 1:
        raise HTTPException(status_code=400, detail="frame_step deve ser >= 1")
    if line_orientation not in {"horizontal", "vertical"}:
        raise HTTPException(status_code=400, detail="line_orientation deve ser 'horizontal' ou 'vertical'")
    if not 0.1 <= line_ratio <= 0.9:
        raise HTTPException(status_code=400, detail="line_ratio deve estar entre 0.1 e 0.9")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Arquivo de video vazio")

    try:
        model, device = load_model(weight_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao carregar modelo: {exc}") from exc

    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_input_path = tmp_input.name
    tmp_output_path = tmp_output.name
    tmp_input.close()
    tmp_output.close()

    try:
        with open(tmp_input_path, "wb") as f:
            f.write(raw)

        cap = cv2.VideoCapture(tmp_input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Arquivo enviado nao e um video valido")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            tmp_output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )

        frame_idx = 0
        cached_count = 0
        cached_points: List[List[float]] = []
        total_line_crossings = 0
        max_frame_count = 0
        line_pos = int((frame_height if line_orientation == "horizontal" else frame_width) * line_ratio)
        tracks: Dict[int, Dict[str, Any]] = {}
        next_track_id = 1
        max_track_gap = 15
        match_distance = 40.0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb).convert("RGB")

            resized_img, points, count = run_inference(model, device, frame_pil, threshold)
            if resized_img.size != (frame_width, frame_height):
                # Mantem dimensoes do video de saida iguais ao video de entrada.
                resized_width, resized_height = resized_img.size
                resized_img = resized_img.resize((frame_width, frame_height))
                scale_x = frame_width / max(1, resized_width)
                scale_y = frame_height / max(1, resized_height)
                points = [[p[0] * scale_x, p[1] * scale_y] for p in points]

            if frame_idx % frame_step == 0:
                cached_count = count
                cached_points = points
                max_frame_count = max(max_frame_count, cached_count)

                assigned_tracks = set()
                updated_tracks: Dict[int, Dict[str, Any]] = {}
                for point in cached_points:
                    px, py = point[0], point[1]
                    best_track_id = None
                    best_distance = float("inf")
                    for track_id, track in tracks.items():
                        if track_id in assigned_tracks:
                            continue
                        if frame_idx - track["last_seen"] > max_track_gap:
                            continue
                        tx, ty = track["pos"]
                        dist = ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5
                        if dist < best_distance and dist <= match_distance:
                            best_distance = dist
                            best_track_id = track_id

                    if best_track_id is None:
                        best_track_id = next_track_id
                        next_track_id += 1
                        updated_tracks[best_track_id] = {
                            "pos": (px, py),
                            "last_seen": frame_idx,
                            "crossed": False,
                        }
                    else:
                        prev_x, prev_y = tracks[best_track_id]["pos"]
                        crossed = tracks[best_track_id]["crossed"]
                        if line_orientation == "horizontal":
                            crossed_line = (prev_y < line_pos <= py) or (prev_y > line_pos >= py)
                        else:
                            crossed_line = (prev_x < line_pos <= px) or (prev_x > line_pos >= px)
                        if crossed_line and not crossed:
                            total_line_crossings += 1
                            crossed = True
                        updated_tracks[best_track_id] = {
                            "pos": (px, py),
                            "last_seen": frame_idx,
                            "crossed": crossed,
                        }
                    assigned_tracks.add(best_track_id)

                for track_id, track in tracks.items():
                    if track_id not in updated_tracks and frame_idx - track["last_seen"] <= max_track_gap:
                        updated_tracks[track_id] = track
                tracks = updated_tracks

            annotated_frame = draw_points_on_image(resized_img, cached_points)
            if line_orientation == "horizontal":
                cv2.line(annotated_frame, (0, line_pos), (frame_width, line_pos), (255, 0, 0), 2)
            else:
                cv2.line(annotated_frame, (line_pos, 0), (line_pos, frame_height), (255, 0, 0), 2)
            cv2.putText(
                annotated_frame,
                f"Pessoas no frame: {cached_count}",
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                f"Total que cruzaram a linha: {total_line_crossings}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated_frame)
            frame_idx += 1

        cap.release()
        writer.release()

    except HTTPException:
        if os.path.exists(tmp_input_path):
            os.remove(tmp_input_path)
        raise
    except Exception as exc:
        if os.path.exists(tmp_input_path):
            os.remove(tmp_input_path)
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)
        raise HTTPException(status_code=500, detail=f"Falha ao processar video: {exc}") from exc
    finally:
        if os.path.exists(tmp_input_path):
            os.remove(tmp_input_path)

    filename_base = os.path.splitext(file.filename or "video")[0]
    return FileResponse(
        path=tmp_output_path,
        media_type="video/mp4",
        filename=f"{filename_base}_pred.mp4",
        headers={
            "X-Line-Crossing-Total": str(total_line_crossings),
            "X-Max-Frame-Count": str(max_frame_count),
        },
        background=BackgroundTask(lambda: os.remove(tmp_output_path) if os.path.exists(tmp_output_path) else None),
    )
