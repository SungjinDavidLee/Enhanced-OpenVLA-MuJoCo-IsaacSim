#!/usr/bin/env python3
"""브리지에 OpenVLA-OFT 정책 경로를 추가한다 (기본 경로는 그대로).

  --policy-kind oft              OFT 서버(/predict_chunk) 사용.  기본 openvla
  --oft-open-loop-steps 8        받은 청크에서 실행할 스텝 수 (OFT LIBERO 평가 기본 8)
  --oft-proprio-calib PATH       right_gripper → robosuite eef 프레임 보정 JSON (oft_proprio_calib.py 가 만듦)

OFTClient.predict(rgb, instruction) 는 OpenVLAClient 와 같은 모양으로 한 스텝 액션을 돌려주고,
큐가 비었을 때만 서버에 agentview · 손목 · proprio 를 보낸다.  나머지 경로(액션 규약, 어댑터, IK,
캡처, S0-S10 판정)는 OpenVLA 와 동일하다.

    python3 patch_bridge_oft_client.py ~/isaac_libero/bridge/run_openvla_isaac_bridge.py
"""

import shutil
import sys
from pathlib import Path

CLASS_ANCHOR = "class OpenVLAClient:\n"
CLASS_CODE = '''def _oft_mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = 0.25 * s; q[j] = (R[j, i] + R[i, j]) / s; q[k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def _oft_quat2axisangle(q_xyzw: np.ndarray) -> np.ndarray:
    """robosuite.utils.transform_utils.quat2axisangle 와 같은 식 (부호 정규화 없음)."""
    q = np.asarray(q_xyzw, dtype=np.float64).copy()
    q[3] = float(np.clip(q[3], -1.0, 1.0))
    den = float(np.sqrt(1.0 - q[3] * q[3]))
    if den < 1e-9:
        return np.zeros(3)
    return q[:3] * 2.0 * float(np.arccos(q[3])) / den


class OFTProprio:
    """Isaac 상태 → OFT(LIBERO) proprio 8: robot0_eef_pos 3 · quat2axisangle(robot0_eef_quat) 3 · gripper_qpos 2."""

    def __init__(self, bridge: Any, calib_path: str):
        calib = json.loads(Path(calib_path).expanduser().read_text())
        self.bridge = bridge
        self.R_off = np.asarray(calib["R_off"], dtype=np.float64).reshape(3, 3)       # R_rg = R_obs @ R_off
        self.p_local = np.asarray(calib["p_local_m"], dtype=np.float64).reshape(3)     # p_obs = p_rg + R_rg @ p_local
        self.prev_q = np.asarray(calib["q_ref_xyzw"], dtype=np.float64).reshape(4)     # MuJoCo xquat 부호 연속성 기준
        LOGGER.info("[OFT Proprio] calib=%s R_off spread=%.2f deg p_local=%s", calib_path,
                    float(calib.get("R_off_spread_deg", float("nan"))), np.round(self.p_local, 4).tolist())

    def __call__(self) -> np.ndarray:
        pose = self.bridge.get_current_ee_pose()
        R_rg = np.asarray(pose.rotation, dtype=np.float64).reshape(3, 3)
        p_rg = np.asarray(pose.position, dtype=np.float64).reshape(3)
        R_obs = R_rg @ self.R_off.T
        p_obs = p_rg + R_rg @ self.p_local - np.asarray(getattr(self.bridge, "libero_world_offset", np.zeros(3)), dtype=np.float64)
        q = _oft_mat_to_quat_xyzw(R_obs)
        if float(np.dot(q, self.prev_q)) < 0.0:
            q = -q
        self.prev_q = q
        joints = np.asarray(self.bridge.robot.get_joint_positions(), dtype=np.float64).reshape(-1)
        fi = list(getattr(self.bridge, "finger_indices", [7, 8]))
        grip = np.array([joints[fi[0]], -joints[fi[1]]], dtype=np.float64)
        return np.concatenate([p_obs, _oft_quat2axisangle(q), grip]).astype(np.float32)


class OFTClient:
    """OpenVLAClient 와 같은 predict() 모양.  큐가 비면 /predict_chunk 로 8 스텝을 받아 한 스텝씩 내준다."""

    def __init__(self, base_url: str, timeout_s: float, open_loop_steps: int = 8):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.open_loop_steps = max(1, int(open_loop_steps))
        self.queue: list[np.ndarray] = []
        self.last_policy_rgb: np.ndarray | None = None
        self.state_fn = None
        self.wrist_fn = None
        self.policy_crop = None
        self.queries = 0

    def attach(self, state_fn: Any, wrist_fn: Any) -> None:
        self.state_fn, self.wrist_fn = state_fn, wrist_fn

    def health(self) -> dict[str, Any]:
        with urllib.request.urlopen(self.base_url + "/health", timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _png_b64(rgb: np.ndarray) -> str:
        import cv2
        ok, enc = cv2.imencode(".png", cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("could not encode frame as PNG")
        return base64.b64encode(enc.tobytes()).decode("ascii")

    def predict(self, rgb: np.ndarray, instruction: str) -> tuple[np.ndarray, np.ndarray]:
        if not self.queue:
            import cv2
            wrist = self.wrist_fn() if self.wrist_fn is not None else None
            if wrist is None:
                raise RuntimeError("OFT needs the wrist camera: set wrist_camera.enabled in the yaml (use the _oft yaml)")
            state = self.state_fn() if self.state_fn is not None else None
            if state is None or np.asarray(state).size != 8:
                raise RuntimeError("OFT needs an 8-D proprio state (--oft-proprio-calib)")
            payload = json.dumps({"instruction": instruction, "image_base64": self._png_b64(rgb),
                                  "wrist_base64": self._png_b64(wrist),
                                  "state": [float(v) for v in np.asarray(state).reshape(-1)]}).encode("utf-8")
            request = urllib.request.Request(self.base_url + "/predict_chunk", data=payload,
                                             headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                raise RuntimeError(f"OFT server HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}") from error
            actions = result.get("actions") or []
            if not bool(result.get("ok")) or not actions:
                raise RuntimeError(f"invalid OFT response: {str(result)[:300]}")
            self.queue = [np.asarray(a, dtype=np.float64).reshape(-1)[:7] for a in actions[: self.open_loop_steps]]
            prgb = cv2.imdecode(np.frombuffer(base64.b64decode(result["policy_rgb_base64"]), np.uint8), cv2.IMREAD_COLOR)
            self.last_policy_rgb = cv2.cvtColor(prgb, cv2.COLOR_BGR2RGB)
            self.queries += 1
            if self.queries <= 2:
                LOGGER.info("[OFT] query %d state=%s first=%s", self.queries,
                            np.round(np.asarray(state), 4).tolist(), np.round(self.queue[0], 4).tolist())
        action = self.queue.pop(0)
        return action, self.last_policy_rgb


'''

ARG_ANCHOR = '''    parser.add_argument("--server-url", default="http://127.0.0.1:8000")\n'''
ARG_NEW = '''    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--policy-kind", choices=["openvla", "oft"], default="openvla",
                        help="oft: OpenVLA-OFT server (/predict_chunk) with wrist + proprio, chunked actions")
    parser.add_argument("--oft-open-loop-steps", type=int, default=8)
    parser.add_argument("--oft-proprio-calib", default="/home/user/isaac_libero/bridge/oft_proprio_calib.json")
'''

BUILD_ANCHOR = '''        elif args.mock_action is None:
            client = OpenVLAClient(
'''
BUILD_NEW = '''        elif args.mock_action is None and getattr(args, "policy_kind", "openvla") == "oft":
            client = OFTClient(args.server_url, args.request_timeout_s, args.oft_open_loop_steps)
            client.attach(state_fn=OFTProprio(bridge, args.oft_proprio_calib),
                          wrist_fn=getattr(scene, "get_wrist_rgb", None))
            server_health = client.health()
            LOGGER.info("OFT health: %s", server_health)
            if server_health.get("kind") != "oft":
                raise RuntimeError(f"--policy-kind oft but the server at {args.server_url} is not an OFT server")
        elif args.mock_action is None:
            client = OpenVLAClient(
'''


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__); return 2
    path = Path(sys.argv[1]).expanduser(); text = path.read_text()
    if "class OFTClient:" in text:
        print("이미 적용됨:", path); return 0
    for name, old in (("class", CLASS_ANCHOR), ("argparse", ARG_ANCHOR), ("client 생성", BUILD_ANCHOR)):
        if text.count(old) != 1:
            print(f"기준 문자열({name}) 이 1곳이 아님: {text.count(old)}"); return 1
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak_oft"))
    text = text.replace(CLASS_ANCHOR, CLASS_CODE + CLASS_ANCHOR, 1).replace(ARG_ANCHOR, ARG_NEW, 1).replace(BUILD_ANCHOR, BUILD_NEW, 1)
    path.write_text(text)
    print("적용 완료:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
