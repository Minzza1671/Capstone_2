import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

P2PNET_DIR = Path(__file__).resolve().parents[2] / "p2pnet"


def _insert_p2pnet_path():
    p = str(P2PNET_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _round128(w: int, h: int) -> Tuple[int, int]:
    return max(128, w // 128 * 128), max(128, h // 128 * 128)


def _build_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class P2PNetCounter:
    def __init__(self, weight_path: str, threshold: float = 0.50,
                 backbone: str = "vgg16_bn", row: int = 2, line: int = 2,
                 device: str = "cpu"):
        _insert_p2pnet_path()
        from models import build_model

        class _Args:
            pass

        a = _Args()
        a.backbone = backbone
        a.row = row
        a.line = line

        self.model = build_model(a, training=False)

        try:
            ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(weight_path), map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state)

        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.threshold = threshold
        self.transform = _build_transform()

    @torch.no_grad()
    def predict(self, frame_bgr) -> List[Tuple[float, float, float]]:
        """Returns list of (x, y, score) in original frame coordinates."""
        orig_h, orig_w = frame_bgr.shape[:2]
        model_w, model_h = _round128(orig_w, orig_h)

        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if (model_w, model_h) != (orig_w, orig_h):
            img = img.resize((model_w, model_h), Image.LANCZOS
                             if hasattr(Image, "LANCZOS") else Image.ANTIALIAS)

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        out = self.model(tensor)

        scores = torch.nn.functional.softmax(out["pred_logits"], -1)[:, :, 1][0]
        points = out["pred_points"][0]
        mask = scores > self.threshold

        sx = orig_w / max(model_w, 1)
        sy = orig_h / max(model_h, 1)

        result = []
        for p, s in zip(points[mask].cpu().numpy(), scores[mask].cpu().numpy()):
            x, y = float(p[0]) * sx, float(p[1]) * sy
            if 0 <= x < orig_w and 0 <= y < orig_h:
                result.append((x, y, float(s)))
        return result

    def count_in_roi(
        self, frame_bgr, roi_manager
    ) -> Tuple[int, List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """Returns (roi_count, all_points, roi_points)."""
        all_pts = self.predict(frame_bgr)
        inside = [p for p in all_pts if roi_manager.contains_point((int(p[0]), int(p[1])))]
        return len(inside), all_pts, inside
