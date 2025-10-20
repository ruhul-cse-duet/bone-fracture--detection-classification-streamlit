import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


_cached_models = {
    'yolo': None,
    'classifier': None,
}


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yolo_model() -> YOLO:
    if _cached_models['yolo'] is not None:
        return _cached_models['yolo']
    weights_path = _get_project_root() / 'models' / 'yolov11_trained.pt'
    model = YOLO(str(weights_path))
    _cached_models['yolo'] = model
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


def run_detection_and_classification(pil_image: Image.Image, conf_threshold: float = 0.25):
    yolo_model = _load_yolo_model()
    clf_model = _load_classifier_model()

    # Ensure RGB
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Run YOLO detection
    results = yolo_model.predict(source=np.array(pil_image), conf=conf_threshold, verbose=False)

    annotated = np.array(pil_image).copy()
    detections = []
    if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
        for b in results[0].boxes:
            xyxy = b.xyxy.cpu().numpy().astype(int)[0]
            score = float(b.conf.cpu().numpy()[0]) if b.conf is not None else 0.0
            cls_id = int(b.cls.cpu().numpy()[0]) if b.cls is not None else -1
            detections.append({'bbox': xyxy.tolist(), 'score': score, 'class_id': cls_id})
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{score:.2f}"
            cv2.putText(annotated, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Classification
    image_tensor = _inference_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = float(clf_model(image_tensor).item())
    # Binary label: 1 -> Fractured, 0 -> Non-Fractured
    threshold = 0.5
    predicted = 1 if prob >= threshold else 0
    confidence = prob if predicted == 1 else 1.0 - prob

    return annotated, predicted, confidence, detections


def prediction_img(pil_image: Image.Image):
    # Backward-compat wrapper used by existing app code (will be updated)
    annotated, predicted, confidence, detections = run_detection_and_classification(pil_image)
    return annotated, predicted, confidence
