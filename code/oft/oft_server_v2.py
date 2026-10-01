#!/usr/bin/env python3
"""OpenVLA-OFT 추론 서버 (Isaac 브리지용, 청크 출력).

openvla-oft 환경에서 실행한다.  입력은 agentview · 손목 RGB · proprio 8, 출력은 8 스텝 액션 묶음이다.
OFT 공식 LIBERO 평가(run_libero_eval.py) 의 전처리를 따른다.

  agentview   Isaac 원본 → 정사각 크롭 → 면적 --crop-area 크롭 (LIBERO 화각 맞춤) → 256 → resize_image_for_policy(224)
              OFT 내부가 다시 center_crop(0.9) 을 하므로, OpenVLA 서버의 0.35 에서 0.9 를 뺀 0.389 가 기본값
  손목        LIBERO 규격 손목 카메라(256², fovy 75°) 원본 → resize_image_for_policy(224).  크롭 없음
  proprio     EE 위치 3 · 축각 3 · 손가락 2 (robosuite 규약, 브리지가 만들어 보냄)
  출력        get_vla_action 의 8 × 7 (정규화 해제된 원시값; 집게 1 = 열림, OpenVLA 서버와 같은 규약)

    conda activate openvla-oft
    python oft_server.py --checkpoint ~/openvla_ckpt/oft_libero_spatial --port 8001

    GET  /health          {"ok": true, "kind": "oft", "chunk": 8, ...}
    POST /predict_chunk   {"instruction", "image_base64", "wrist_base64", "state": [8], ("crop_area")} → {"ok", "actions": [[7]]*8, "policy_rgb_base64"}
                          crop_area 를 주면 그 요청만 다른 크롭 (MuJoCo 256 화면 검증은 1.0)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
from PIL import Image

LOGGER = logging.getLogger("OFTServer")


def decode_png_b64(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


def encode_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def square_area_crop(img: Image.Image, area: float, size: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    if area < 1.0:
        f = math.sqrt(area)
        w, h = img.size
        cw, ch = max(1, int(round(w * f))), max(1, int(round(h * f)))
        left, top = (w - cw) // 2, (h - ch) // 2
        img = img.crop((left, top, left + cw, top + ch))
    return img.resize((size, size), Image.Resampling.BILINEAR)


class OFTPolicy:
    def __init__(self, a):
        from experiments.robot import openvla_utils as ou
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

        self.ou = ou
        self.chunk = int(NUM_ACTIONS_CHUNK)
        self.cfg = SimpleNamespace(
            model_family="openvla", pretrained_checkpoint=os.path.expanduser(a.checkpoint),
            use_l1_regression=True, use_diffusion=False, num_diffusion_steps=50,
            num_diffusion_steps_train=50, num_diffusion_steps_inference=50, use_film=False,
            num_images_in_input=2, use_proprio=True, center_crop=True, num_open_loop_steps=self.chunk,
            lora_rank=32, unnorm_key=a.unnorm_key, load_in_8bit=False, load_in_4bit=False,
            task_suite_name="libero_spatial", seed=7, use_relative_actions=False,
        )
        t0 = time.time()
        self.vla = ou.get_vla(self.cfg)
        self.processor = ou.get_processor(self.cfg)
        self.action_head = ou.get_action_head(self.cfg, self.vla.llm_dim)
        self.proprio_projector = ou.get_proprio_projector(self.cfg, self.vla.llm_dim, proprio_dim=PROPRIO_DIM)
        stats = getattr(self.vla, "norm_stats", {}) or {}
        if self.cfg.unnorm_key not in stats:
            alt = self.cfg.unnorm_key + "_no_noops"
            if alt in stats:
                self.cfg.unnorm_key = alt
            else:
                raise RuntimeError(f"unnorm_key {self.cfg.unnorm_key} 없음. 있는 키: {list(stats)}")
        self.resize = getattr(ou, "resize_image_for_policy", None)
        self.crop_area = float(a.crop_area)
        self.lock = threading.Lock()
        LOGGER.info("OFT ready in %.0fs: chunk=%d proprio=%d unnorm_key=%s crop_area=%.3f resize=%s",
                    time.time() - t0, self.chunk, PROPRIO_DIM, self.cfg.unnorm_key, self.crop_area,
                    "resize_image_for_policy" if self.resize else "PIL")

    def _to_policy(self, img256: Image.Image) -> np.ndarray:
        arr = np.asarray(img256.convert("RGB"), dtype=np.uint8)
        if self.resize is not None:
            return np.asarray(self.resize(arr, 224), dtype=np.uint8)
        return np.asarray(img256.resize((224, 224), Image.Resampling.LANCZOS), dtype=np.uint8)

    def predict(self, instruction: str, agent: Image.Image, wrist: Image.Image, state, crop_area: float | None = None) -> tuple[list, Image.Image]:
        agent256 = square_area_crop(agent, self.crop_area if crop_area is None else float(crop_area), 256)
        wrist256 = square_area_crop(wrist, 1.0, 256)
        obs = {"full_image": self._to_policy(agent256), "wrist_image": self._to_policy(wrist256),
               "state": np.asarray(state, dtype=np.float32).reshape(-1)}
        with self.lock:
            actions = self.ou.get_vla_action(self.cfg, self.vla, self.processor, obs, instruction,
                                             action_head=self.action_head, proprio_projector=self.proprio_projector,
                                             use_film=False)
        acts = [np.asarray(x, dtype=np.float64).reshape(-1)[:7].tolist() for x in actions]
        return acts, Image.fromarray(obs["full_image"])


def make_handler(policy: OFTPolicy):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            LOGGER.debug("%s - %s", self.address_string(), fmt % args)

        def do_GET(self):
            if self.path.startswith("/health"):
                self._send(200, {"ok": True, "kind": "oft", "chunk": policy.chunk,
                                 "unnorm_key": policy.cfg.unnorm_key, "crop_area": policy.crop_area})
            else:
                self._send(404, {"ok": False})

        def do_POST(self):
            if not self.path.startswith("/predict_chunk"):
                self._send(404, {"ok": False}); return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
                t0 = time.perf_counter()
                acts, prgb = policy.predict(str(req["instruction"]), decode_png_b64(req["image_base64"]),
                                            decode_png_b64(req["wrist_base64"]), req["state"], req.get("crop_area"))
                self._send(200, {"ok": True, "actions": acts, "policy_rgb_base64": encode_png_b64(prgb),
                                 "inference_ms": 1000.0 * (time.perf_counter() - t0)})
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("predict_chunk failed")
                self._send(500, {"ok": False, "error": repr(exc)})
    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="~/openvla_ckpt/oft_libero_spatial")
    ap.add_argument("--unnorm-key", default="libero_spatial_no_noops")
    ap.add_argument("--crop-area", type=float, default=0.35 / 0.9,
                    help="agentview 면적 크롭 (OFT 내부 0.9 크롭 전).  OpenVLA 서버 0.35 와 같은 최종 화각")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    policy = OFTPolicy(a)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(policy))
    LOGGER.info("listening on http://%s:%d", a.host, a.port)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
