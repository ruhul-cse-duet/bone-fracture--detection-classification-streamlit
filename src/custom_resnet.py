import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_cached_models = {
    'yolo': None,
    'classifier': None,
}

def _ensure_cv2_headless():
    try:
        import cv2  # noqa
    except Exception as e:
        raise ImportError(
            "cv2 import failed. Ensure `opencv-python-headless` is in requirements.txt "
            "and there is no conflicting opencv installed. Original: " + str(e)
        )


# ---------- Helper Functions ----------

def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yolo_model():
    weights_path = Path("models/yolov11_trained.pt")
    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        import torch
        print("⚠️ GPU model load failed, retrying on CPU:", e)
        try:
            ckpt = torch.load(weights_path, map_location='cpu')
            model = YOLO()
            if 'model' in ckpt:
                model.model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
            else:
                model.model.load_state_dict(ckpt, strict=False)
        except Exception as inner_e:
            print("❌ Failed to load YOLO model:", inner_e)
            raise RuntimeError("YOLO model could not be loaded in CPU mode.")
    return model



def _load_classifier_model() -> torch.nn.Module:
    if _cached_models['classifier'] is not None:
        return _cached_models['classifier']
    model = models.efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 1),
        nn.Sigmoid(),
    )
    model.to(device)
    weights_path = _get_project_root() / 'models' / 'best_model_efficient.pth'
    state = torch.load(str(weights_path), map_location=device)
    try:
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    except Exception:
        pass
    model.eval()
    _cached_models['classifier'] = model
    return model


_inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------- Core Function ----------

def run_detection_and_classification(pil_image: Image.Image, conf_threshold: float = 0.25):
    yolo_model = _load_yolo_model()
    clf_model = _load_classifier_model()

    # Ensure RGB
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # YOLO Detection
    results = yolo_model.predict(source=np.array(pil_image), conf=conf_threshold, verbose=False)

    annotated = pil_image.copy()
    draw = ImageDraw.Draw(annotated)
    detections = []

    if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
        for b in results[0].boxes:
            xyxy = b.xyxy.cpu().numpy().astype(int)[0]
            score = float(b.conf.cpu().numpy()[0]) if b.conf is not None else 0.0
            cls_id = int(b.cls.cpu().numpy()[0]) if b.cls is not None else -1

            detections.append({'bbox': xyxy.tolist(), 'score': score, 'class_id': cls_id})
            x1, y1, x2, y2 = xyxy

            # ✅ Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

            # ✅ Draw text label
            label = f"{score:.2f}"
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            # ✅ Fix: Pillow 10+ uses textbbox instead of textsize
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                text_width, text_height = font.getsize(label)

            text_bg = [x1, max(0, y1 - text_height - 4), x1 + text_width + 4, y1]
            draw.rectangle(text_bg, fill=(0, 255, 0))
            draw.text((x1 + 2, max(0, y1 - text_height - 2)), label, fill=(255, 255, 255), font=font)

    # Classification
    image_tensor = _inference_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = float(clf_model(image_tensor).item())

    # Binary label: 1 -> Fractured, 0 -> Non-Fractured
    threshold = 0.5
    predicted = 1 if prob >= threshold else 0
    confidence = prob if predicted == 1 else 1.0 - prob

    annotated_np = np.array(annotated)
    return annotated_np, predicted, confidence, detections


def prediction_img(pil_image: Image.Image):
    annotated, predicted, confidence, detections = run_detection_and_classification(pil_image)
    return annotated, predicted, confidence
