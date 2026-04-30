import io
import os
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as standard_transforms
from PIL import Image

from models import build_model


DEFAULT_WEIGHT_PATH = "./weights/SHTechA.pth"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource(show_spinner=False)
def load_model(weight_path: str):
    device = get_device()
    args = SimpleNamespace(backbone="vgg16_bn", row=2, line=2)
    model = build_model(args)
    model.to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device


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


def run_inference(model, device, img_raw: Image.Image, threshold: float):
    resized_img, img_tensor = preprocess_image(img_raw)
    samples = torch.Tensor(img_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
        outputs_points = outputs["pred_points"][0]

    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    img_to_draw = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), 2, (57, 255, 20), -1)

    img_annotated = cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB)
    return resized_img, img_annotated, predict_cnt


def process_video(
    model,
    device,
    uploaded_video,
    threshold: float,
    frame_step: int,
    line_orientation: str,
    line_ratio: float,
):
    video_bytes_input = uploaded_video.getvalue()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(video_bytes_input)
        input_path = input_tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_tmp:
        output_path = output_tmp.name

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Arquivo de video invalido.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )

        frame_idx = 0
        cached_count = 0
        cached_points = []
        total_line_crossings = 0
        max_frame_count = 0
        line_pos = int((frame_height if line_orientation == "horizontal" else frame_width) * line_ratio)
        tracks = {}
        next_track_id = 1
        max_track_gap = 15
        match_distance = 40.0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb).convert("RGB")
            resized_img, _, count = run_inference(model, device, frame_pil, threshold)

            resized_w, resized_h = resized_img.size
            scale_x = frame_width / max(1, resized_w)
            scale_y = frame_height / max(1, resized_h)
            resized_img_arr = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)
            if (resized_w, resized_h) != (frame_width, frame_height):
                resized_img_arr = cv2.resize(resized_img_arr, (frame_width, frame_height))

            points = []
            if frame_idx % frame_step == 0:
                resized_img2, img_tensor = preprocess_image(frame_pil)
                samples = torch.Tensor(img_tensor).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(samples)
                    outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
                    outputs_points = outputs["pred_points"][0]
                points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
                cached_points = [[p[0] * scale_x, p[1] * scale_y] for p in points]
                cached_count = count
                max_frame_count = max(max_frame_count, cached_count)

                assigned_tracks = set()
                updated_tracks = {}
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

            for p in cached_points:
                resized_img_arr = cv2.circle(
                    resized_img_arr, (int(p[0]), int(p[1])), 2, (57, 255, 20), -1
                )
            if line_orientation == "horizontal":
                cv2.line(resized_img_arr, (0, line_pos), (frame_width, line_pos), (255, 0, 0), 2)
            else:
                cv2.line(resized_img_arr, (line_pos, 0), (line_pos, frame_height), (255, 0, 0), 2)

            cv2.putText(
                resized_img_arr,
                f"Pessoas no frame: {cached_count}",
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                resized_img_arr,
                f"Total que cruzaram a linha: {total_line_crossings}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            writer.write(resized_img_arr)
            frame_idx += 1

        cap.release()
        writer.release()

        with open(output_path, "rb") as f:
            video_bytes = f.read()
        return {
            "video_bytes": video_bytes,
            "total_line_crossings": total_line_crossings,
            "max_frame_count": max_frame_count,
        }
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


def build_video_preview(uploaded_video, line_orientation: str, line_ratio: float, frame_ratio: float = 0.1):
    video_bytes_input = uploaded_video.getvalue()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(video_bytes_input)
        input_path = input_tmp.name

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Nao foi possivel abrir o video para preview.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        target_frame = min(total_frames - 1, max(0, int(total_frames * frame_ratio)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Nao foi possivel ler o frame de preview.")

        frame_height, frame_width = frame_bgr.shape[:2]
        line_pos = int((frame_height if line_orientation == "horizontal" else frame_width) * line_ratio)
        if line_orientation == "horizontal":
            cv2.line(frame_bgr, (0, line_pos), (frame_width, line_pos), (255, 0, 0), 2)
        else:
            cv2.line(frame_bgr, (line_pos, 0), (line_pos, frame_height), (255, 0, 0), 2)

        preview_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return preview_rgb
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


st.set_page_config(page_title="P2PNet - Contagem de Pessoas", layout="wide")
st.title("Sistema de Contagem de Pessoas")
st.write("Selecione o tipo de entrada (imagem ou video), ajuste os parametros e execute.")

tab_img, tab_video = st.tabs(["Imagem", "Video"])

with tab_img:
    threshold_img = st.slider("Confianca", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    uploaded_image = st.file_uploader("Selecione uma imagem", type=["jpg", "jpeg", "png"])
    run_image = st.button("Executar contagem na imagem")

    if run_image:
        if uploaded_image is None:
            st.error("Envie uma imagem antes de executar.")
        else:
            try:
                model, device = load_model(DEFAULT_WEIGHT_PATH)
                img_raw = Image.open(uploaded_image).convert("RGB")
                original_img, annotated_img, count = run_inference(model, device, img_raw, threshold_img)

                st.success(f"Contagem estimada: {count} pessoas")
                st.caption(f"Dispositivo de execucao: {device}")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Imagem processada")
                    st.image(original_img, use_container_width=True)
                with col2:
                    st.subheader("Resultado com pontos")
                    st.image(annotated_img, use_container_width=True)

                buffer = io.BytesIO()
                Image.fromarray(annotated_img).save(buffer, format="JPEG")
                st.download_button(
                    label="Baixar imagem com contagem",
                    data=buffer.getvalue(),
                    file_name=f"resultado_{count}.jpg",
                    mime="image/jpeg",
                )
            except Exception as exc:
                st.error(f"Falha ao executar inferencia: {exc}")

with tab_video:
    threshold_video = st.slider(
        "Confianca (video)", min_value=0.10, max_value=0.90, value=0.50, step=0.05
    )
    frame_step = st.number_input(
        "Processar um frame a cada N frames",
        min_value=1,
        max_value=30,
        value=1,
        step=1,
    )
    uploaded_video = st.file_uploader("Selecione um video", type=["mp4", "mov", "avi", "mkv"])
    line_orientation_ui = "horizontal"
    line_ratio_ui = 0.6
    preview_frame_ratio = 0.1

    if uploaded_video is not None:
        try:
            preview_col, controls_col = st.columns([3, 2])
            with controls_col:
                st.subheader("Ajustes da linha")
                line_orientation_ui = st.selectbox(
                    "Orientacao da linha de contagem",
                    options=["horizontal", "vertical"],
                    index=0,
                )
                line_ratio_ui = st.slider(
                    "Posicao da linha (proporcao)",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.6,
                    step=0.05,
                )
                preview_frame_ratio = st.slider(
                    "Frame para preview (proporcao do video)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                )

            preview_img = build_video_preview(
                uploaded_video, line_orientation_ui, float(line_ratio_ui), float(preview_frame_ratio)
            )
            with preview_col:
                st.subheader("Preview da linha no video")
                st.image(
                    preview_img,
                    caption="A linha azul indica a posicao de contagem",
                    use_container_width=True,
                )
        except Exception as exc:
            st.warning(f"Nao foi possivel gerar preview: {exc}")

    run_video = st.button("Executar contagem no video")

    if run_video:
        if uploaded_video is None:
            st.error("Envie um video antes de executar.")
        else:
            try:
                model, device = load_model(DEFAULT_WEIGHT_PATH)
                with st.spinner("Processando video, isso pode levar alguns minutos..."):
                    result = process_video(
                        model,
                        device,
                        uploaded_video,
                        threshold_video,
                        int(frame_step),
                        line_orientation_ui,
                        float(line_ratio_ui),
                    )
                    video_bytes = result["video_bytes"]

                st.success("Video processado com sucesso.")
                st.caption(f"Dispositivo de execucao: {device}")
                st.metric("Total que cruzaram a linha", int(result["total_line_crossings"]))
                st.metric("Pico de pessoas por frame", int(result["max_frame_count"]))
                st.video(video_bytes)
                st.download_button(
                    label="Baixar video com contagem",
                    data=video_bytes,
                    file_name="resultado_video.mp4",
                    mime="video/mp4",
                )
            except Exception as exc:
                st.error(f"Falha ao processar video: {exc}")
