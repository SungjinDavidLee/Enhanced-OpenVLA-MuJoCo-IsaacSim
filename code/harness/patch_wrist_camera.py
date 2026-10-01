#!/usr/bin/env python3
"""손목 D405 카메라를 환경·브리지·yaml 에 영구 추가한다.

  env    _create_wrist_d405(): /World/Franka/panda_hand 아래에 RtxCamera (손 프레임 기준 위치·기울기, D405 광학)
         get_wrist_rgb():      스텝마다 손목 RGB
  bridge 캡처 클래스에 "wrist_rgb" 추가  → wrist_rgb.mp4 · wrist_rgb_first.png · wrist_rgb_last.png
         capture_raw 직후 손목 프레임도 기록 → raw_rgb 와 프레임 수가 같음
  yaml   wrist_camera 블록 추가 (enabled 로 켜고 끔)

    python3 patch_wrist_camera.py ~/isaac_libero/bridge

손 프레임(panda_hand): +z 손끝 방향, y 손가락 이동축, x 그 수직.  카메라는 +x 쪽 마운트.
USD 카메라는 자기 −Z 를 보고 +Y 가 화면 위다.  tilt_deg 만큼 손 중심(−x) 쪽으로 기울인다.
"""

import shutil
import sys
from pathlib import Path

ENV_HELPER_ANCHOR = "def set_xform_pose(prim: Usd.Prim, position: Any, quat_wxyz: Any) -> None:"
ENV_HELPER = '''def wrist_camera_quat_wxyz(tilt_deg: float, roll_deg: float) -> list[float]:
    """Camera-to-hand rotation for a hand-mounted camera looking toward the fingertips.

    USD cameras look down their own -Z with +Y up.  In the panda_hand frame +z points
    to the fingertips, so the camera's -Z is aimed along hand +z and tilted by
    ``tilt_deg`` toward hand -x (the hand centre, since the camera sits at +x).
    ``roll_deg`` rotates the image about the viewing axis (180 flips it).
    """
    t = np.radians(float(tilt_deg))
    forward = np.array([-np.sin(t), 0.0, np.cos(t)])          # camera -Z in hand frame
    up = np.array([-np.cos(t), 0.0, -np.sin(t)])              # camera +Y in hand frame (image top toward hand centre)
    z_cam = -forward
    y_cam = up / np.linalg.norm(up)
    x_cam = np.cross(y_cam, z_cam)
    R = np.stack([x_cam, y_cam, z_cam], axis=1)
    r = np.radians(float(roll_deg))
    Rz = np.array([[np.cos(r), -np.sin(r), 0.0], [np.sin(r), np.cos(r), 0.0], [0.0, 0.0, 1.0]])
    R = R @ Rz
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    return [float(v) for v in q]


'''

ENV_BUILD_ANCHOR = "        self._create_agentview_d405()\n"
ENV_BUILD_NEW = "        self._create_agentview_d405()\n        self._create_wrist_d405()\n"

ENV_METHODS_ANCHOR = "    def get_images(self) -> tuple[np.ndarray | None, np.ndarray | None]:"
ENV_METHODS = '''    def _create_wrist_d405(self) -> None:
        """Hand-mounted D405 (config key ``wrist_camera``); created lazily if the hand prim is not there yet."""
        cfg = self.config.get("wrist_camera") or {}
        self.wrist_sensor = getattr(self, "wrist_sensor", None)
        self._wrist_pending = False
        if not bool(cfg.get("enabled", False)) or self.stage is None:
            return
        parent = str(cfg.get("parent_prim", "/World/Franka/panda_hand"))
        if not self.stage.GetPrimAtPath(parent).IsValid():
            self._wrist_pending = True
            print(f"[Camera] wrist_d405 parent {parent} not present yet; will create on first frame")
            return
        path = parent + "/" + str(cfg.get("name", "wrist_d405"))
        old = self.stage.GetPrimAtPath(path)
        if old.IsValid():
            self.stage.RemovePrim(Sdf.Path(path))
        self.wrist_rtx_camera = RtxCamera(path=path, tick_rate=float(cfg.get("frequency", 30.0)))
        prim = self.stage.GetPrimAtPath(path)
        position = [float(v) for v in cfg.get("position_xyz", [0.055, 0.0, 0.030])]
        quat = wrist_camera_quat_wxyz(float(cfg.get("tilt_deg", 15.0)), float(cfg.get("roll_deg", 0.0)))
        set_xform_pose(prim, position, quat)
        focal = float(cfg.get("focal_length_mm", 1.93))
        camera = UsdGeom.Camera(prim)
        camera.GetFocalLengthAttr().Set(focal)
        camera.GetHorizontalApertureAttr().Set(aperture_from_fov(focal, float(cfg.get("horizontal_fov_deg", 87.0))))
        camera.GetVerticalApertureAttr().Set(aperture_from_fov(focal, float(cfg.get("vertical_fov_deg", 58.0))))
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(float(cfg.get("depth_min_m", 0.02)), float(cfg.get("render_clip_far_m", 5.0))))
        width, height = [int(v) for v in cfg.get("resolution", [640, 360])]
        self.wrist_sensor = CameraSensor(self.wrist_rtx_camera, resolution=(height, width), annotators=["rgb"])
        print(f"[Camera] wrist_d405 path={path} position={position} tilt={cfg.get('tilt_deg', 15.0)} "
              f"roll={cfg.get('roll_deg', 0.0)} fov={cfg.get('horizontal_fov_deg', 87.0)}x{cfg.get('vertical_fov_deg', 58.0)} res={width}x{height}")

    def get_wrist_rgb(self) -> np.ndarray | None:
        if getattr(self, "_wrist_pending", False):
            self._create_wrist_d405()
        sensor = getattr(self, "wrist_sensor", None)
        if sensor is None:
            return None
        data, _ = sensor.get_data("rgb")
        if hasattr(data, "numpy"):
            data = data.numpy()
        rgb = np.asarray(data)
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            return None
        return np.ascontiguousarray(rgb[..., :3])

'''

BR_COUNTS_OLD = '''            "raw_rgb": 0,
            "openvla_rgb": 0,
            "depth_preview": 0,
        }'''
BR_COUNTS_NEW = '''            "raw_rgb": 0,
            "openvla_rgb": 0,
            "depth_preview": 0,
            "wrist_rgb": 0,
        }'''
BR_FILES_OLD = '''                "raw_rgb": "raw_rgb.mp4",
                "openvla_rgb": "openvla_rgb.mp4",
                "depth_preview": "depth_preview.mp4",
            },'''
BR_FILES_NEW = '''                "raw_rgb": "raw_rgb.mp4",
                "openvla_rgb": "openvla_rgb.mp4",
                "depth_preview": "depth_preview.mp4",
                "wrist_rgb": "wrist_rgb.mp4",
            },'''
BR_METHOD_ANCHOR = "    def capture_policy(self, rgb: np.ndarray, policy_step: int) -> None:"
BR_METHOD = '''    def capture_wrist(self, rgb: np.ndarray | None, policy_step: int) -> None:
        if rgb is None or int(policy_step) % self.capture_every_n_steps != 0:
            return
        self._write("wrist_rgb", rgb, is_rgb=True)

'''
BR_CALL_OLD = '''            if recorder is not None:
                recorder.capture_raw(rgb, depth, step)'''
BR_CALL_NEW = '''            if recorder is not None:
                recorder.capture_raw(rgb, depth, step)
                wrist_rgb = scene.get_wrist_rgb() if hasattr(scene, "get_wrist_rgb") else None
                if wrist_rgb is not None:
                    recorder.capture_wrist(wrist_rgb, step)'''

YAML_BLOCK = '''
# 손목 D405 (patch_wrist_camera.py).  enabled: false 면 생성·기록하지 않는다.
# 손 프레임(panda_hand): +z 손끝, y 손가락 이동축, x 그 수직(카메라 마운트 쪽).
wrist_camera:
  enabled: true
  parent_prim: /World/Franka/panda_hand
  name: wrist_d405
  position_xyz: [0.055, 0.0, 0.030]
  tilt_deg: 15.0
  roll_deg: 0.0
  resolution: [640, 360]
  horizontal_fov_deg: 87.0
  vertical_fov_deg: 58.0
  focal_length_mm: 1.93
  depth_min_m: 0.02
  render_clip_far_m: 5.0
'''


def patch(path: Path, pairs, tag: str) -> bool:
    text = path.read_text()
    if tag in text:
        print("이미 적용됨:", path); return True
    for old, _ in pairs:
        if text.count(old) != 1:
            print("기준 문자열이 1곳이 아님 (%d):\n%s" % (text.count(old), old[:120])); return False
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak_wrist"))
    for old, new in pairs:
        text = text.replace(old, new, 1)
    path.write_text(text); print("적용 완료:", path); return True


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__); return 2
    root = Path(sys.argv[1]).expanduser()
    env = root / "libero_spatial_env_isaac601.py"; br = root / "run_openvla_isaac_bridge.py"; yml = root / "libero_spatial_environment.yaml"
    ok = patch(env, [(ENV_HELPER_ANCHOR, ENV_HELPER + ENV_HELPER_ANCHOR), (ENV_BUILD_ANCHOR, ENV_BUILD_NEW), (ENV_METHODS_ANCHOR, ENV_METHODS + ENV_METHODS_ANCHOR)], "def _create_wrist_d405")
    ok = patch(br, [(BR_COUNTS_OLD, BR_COUNTS_NEW), (BR_FILES_OLD, BR_FILES_NEW), (BR_METHOD_ANCHOR, BR_METHOD + BR_METHOD_ANCHOR), (BR_CALL_OLD, BR_CALL_NEW)], "def capture_wrist") and ok
    y = yml.read_text()
    if "wrist_camera:" in y:
        print("yaml 이미 있음:", yml)
    else:
        shutil.copy2(yml, yml.with_suffix(".yaml.bak_wrist")); yml.write_text(y.rstrip("\n") + "\n" + YAML_BLOCK); print("yaml 블록 추가:", yml)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
