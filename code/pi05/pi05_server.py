#!/usr/bin/env python3
"""π0.5 (openpi pi05_libero) 를 OFT 서버와 같은 방식으로 주고받는 정책 서버.

Isaac Sim 브리지는 고치지 않고 `--policy-kind oft` 로 그대로 쓴다. 브리지 쪽에서 보면 OFT 서버와 똑같다.
    GET  /health          {"ok": true, "kind": "oft", "model": "pi05_libero", "chunk": 5, ...}
    POST /predict_chunk   {"instruction", "image_base64", "wrist_base64", "state": [8], ("crop_area")}
                          → {"ok", "actions": [[7]] * 5, "policy_rgb_base64"}

OFT 서버와 다른 점 (π0.5 의 학습 방식에 맞춤)
  1) 바깥 카메라: 면적 0.389 만큼 가운데를 자름 (= 0.35 ÷ 0.9). π0.5 는 MuJoCo 화면을 자르지 않고 학습했고 내부에서 다시 자르지 않음.
     OFT · OpenVLA 는 서버에서 0.35 로 자른 뒤 모델 안에서 0.9 를 한 번 더 잘라, 결과 시야가 같다.
  2) 손목 카메라: 자르지 않음 (OFT 와 같음). 둘 다 224 로 줄임 (정사각형이라 openpi 의 resize_with_pad 와 같은 결과).
  3) 집게 값: π0.5 는 환경 형식(−1 = 엶, +1 = 닫음)으로 낸다. 브리지는 OFT · OpenVLA 형식(0 = 닫음, 1 = 엶)을 받아
     스스로 환경 형식으로 바꾸므로, 여기서 g = (1 − g_env) / 2 로 바꿔 돌려준다.
  4) 묶음: π0.5 는 10개를 내고, openpi 의 LIBERO 예제처럼 앞 5개만 실행한다 (--chunk 로 바꿀 수 있음).
  5) 서버 시작 때 가짜 입력으로 두 번 미리 계산해 둔다 (첫 계산의 컴파일 시간 때문에 브리지가 시간 초과로 멈추는 것을 막음).
  로봇 자세 8개(손 위치 3 · 축각 3 · 손가락 2)는 OFT 와 같은 형식이라 그대로 넣는다.

실행 (openpi 의 파이썬 환경에서, GPU 메모리를 미리 다 잡지 않게):
    cd ~/Downloads/openpi
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python ~/tsk/VLA-repo_all/scripts/pi05/pi05_server.py --port 8000
"""
import argparse
import base64
import io
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

LOGGER = logging.getLogger("pi05_server")


def decode_png_b64(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


def encode_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()


def square_area_crop(img: Image.Image, area: float, size: int) -> Image.Image:
    """가운데에서 면적 비율 area 인 정사각형을 잘라 size 로 줄인다 (OFT 서버와 같은 방식)."""
    w, h = img.size
    side = int(round(min(w, h) * min(1.0, max(1e-3, area)) ** 0.5))
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)


class Pi05Policy:
    def __init__(self, a):
        from openpi.policies import policy_config
        from openpi.shared import download
        from openpi.training import config as _config
        t0 = time.time()
        self.cfg = _config.get_config(a.config)
        self.ckpt = download.maybe_download(a.checkpoint)
        self.policy = policy_config.create_trained_policy(self.cfg, self.ckpt)
        self.chunk, self.crop_area, self.gripper_out = a.chunk, a.crop_area, a.gripper_out
        self.lock = threading.Lock()
        # 미리 계산: JAX 는 첫 계산 때 GPU 용으로 준비(컴파일)하느라 오래 걸린다. 이걸 서버 시작 때 끝내 두어야
        # 브리지의 첫 요청이 시간 초과로 멈추지 않는다. /health 는 이게 끝난 뒤에야 열린다.
        t1 = time.time()
        dummy = {"observation/image": np.zeros((224, 224, 3), np.uint8), "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
                 "observation/state": np.zeros(8, np.float32), "prompt": "pick up the black bowl and place it on the plate"}
        for _ in range(2):
            self.policy.infer(dummy)
        LOGGER.info("warm-up done in %.0fs", time.time() - t1)
        LOGGER.info("pi05 ready in %.0fs: config=%s ckpt=%s chunk=%d crop_area=%.3f gripper_out=%s",
                    time.time() - t0, a.config, self.ckpt, self.chunk, self.crop_area, self.gripper_out)

    def predict(self, instruction, agent, wrist, state, crop_area=None):
        agent224 = square_area_crop(agent, self.crop_area if crop_area is None else float(crop_area), 224)
        wrist224 = square_area_crop(wrist, 1.0, 224)
        obs = {"observation/image": np.asarray(agent224, dtype=np.uint8),
               "observation/wrist_image": np.asarray(wrist224, dtype=np.uint8),
               "observation/state": np.asarray(state, dtype=np.float32).reshape(-1)[:8],
               "prompt": str(instruction)}
        with self.lock:
            out = self.policy.infer(obs)
        acts = np.asarray(out["actions"], dtype=np.float64)[: self.chunk, :7].copy()
        if self.gripper_out == "openvla":          # 환경 형식(−1 엶, +1 닫음) → 0 닫음, 1 엶
            acts[:, 6] = (1.0 - np.clip(acts[:, 6], -1.0, 1.0)) / 2.0
        return acts.tolist(), agent224


def make_handler(policy: Pi05Policy, a):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/health"):
                self._send(200, {"ok": True, "kind": "oft", "model": "pi05_libero", "config": a.config, "chunk": policy.chunk,
                                 "crop_area": policy.crop_area, "gripper_out": policy.gripper_out,
                                 "gripper_convention": "openvla_libero_0_close_1_open" if policy.gripper_out == "openvla" else "env_minus1_open"})
            else:
                self._send(404, {"ok": False, "error": "unknown path"})

        def do_POST(self):
            if not self.path.startswith("/predict_chunk"):
                return self._send(404, {"ok": False, "error": "unknown path"})
            try:
                n = int(self.headers.get("Content-Length", "0")); req = json.loads(self.rfile.read(n))
                acts, rgb = policy.predict(req["instruction"], decode_png_b64(req["image_base64"]),
                                           decode_png_b64(req["wrist_base64"]), req["state"], req.get("crop_area"))
                self._send(200, {"ok": True, "actions": acts, "policy_rgb_base64": encode_png_b64(rgb)})
            except Exception as e:                      # 서버는 죽지 않고 오류를 돌려준다
                LOGGER.exception("predict failed")
                self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    ap.add_argument("--chunk", type=int, default=5, help="묶음 10개 중 실행할 개수 (openpi LIBERO 예제는 5)")
    ap.add_argument("--crop-area", type=float, default=0.35 / 0.9, help="바깥 카메라 가운데 자르기 면적 비율")
    ap.add_argument("--gripper-out", choices=["openvla", "env"], default="openvla",
                    help="openvla: 0 닫음 · 1 엶 (브리지가 기대하는 형식), env: π0.5 가 낸 그대로")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    policy = Pi05Policy(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(policy, a))
    LOGGER.info("listening on 127.0.0.1:%d", a.port)
    srv.serve_forever()


if __name__ == "__main__":
    main()
