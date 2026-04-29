import io
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
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

    img_annotated = cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB)
    return resized_img, img_annotated, predict_cnt


st.set_page_config(page_title="P2PNet - Contagem de Pessoas", layout="wide")
st.title("Sistema de Contagem de Pessoas")
st.write("Faça upload de uma imagem, ajuste o limiar e clique em executar.")

weight_path = st.text_input("Caminho dos pesos", value=DEFAULT_WEIGHT_PATH)
threshold = st.slider("Confiança", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
uploaded_file = st.file_uploader("Selecione uma imagem", type=["jpg", "jpeg", "png"])
run_button = st.button("Executar contagem")

if run_button:
    if uploaded_file is None:
        st.error("Envie uma imagem antes de executar.")
    else:
        try:
            model, device = load_model(weight_path)
            img_raw = Image.open(uploaded_file).convert("RGB")
            original_img, annotated_img, count = run_inference(model, device, img_raw, threshold)

            st.success(f"Contagem estimada: {count} pessoas")
            st.caption(f"Dispositivo de execução: {device}")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Imagem processada")
                st.image(original_img, use_column_width=True)
            with col2:
                st.subheader("Resultado com pontos")
                st.image(annotated_img, use_column_width=True)

            buffer = io.BytesIO()
            Image.fromarray(annotated_img).save(buffer, format="JPEG")
            st.download_button(
                label="Baixar imagem com contagem",
                data=buffer.getvalue(),
                file_name=f"resultado_{count}.jpg",
                mime="image/jpeg",
            )
        except Exception as exc:
            st.error(f"Falha ao executar inferência: {exc}")
