#!/usr/bin/env python3
"""파지 시험용 물리 설정 yaml 사본을 만든다 (원본 yaml 은 그대로).

    python3 make_physx_variants.py              # 1 차: 반복 계산 · 재질 · 접촉 여유 · 전부
    python3 make_physx_variants.py --torsion    # 2 차: 비틀림 마찰 (px_tor1 · px_tor2 · px_tor1_m6)
    → ~/isaac_libero/bridge/physx_variants/libero_spatial_environment_px_<이름>.yaml (+ _t5)

변형 (그릇 두 개에 같이 적용)
  px_iter   강체 반복 계산 횟수 32 / 4
  px_mat    그릇 물리 재질 마찰 0.95 (MuJoCo 값) · 결합 방식 max, 손가락 재질도 max
  px_off    접촉 여유 2 mm · 멈춤 간격 0
  px_all    위 셋 + 질량 6 g
"""

from pathlib import Path

import yaml

BR = Path.home() / "isaac_libero/bridge"
OUT = BR / "physx_variants"
BOWLS = ("target_bowl", "distractor_bowl")
VARIANTS_TORSION = {
    "px_tor1": {"torsional_patch_radius_m": 0.01, "min_torsional_patch_radius_m": 0.005},
    "px_tor2": {"torsional_patch_radius_m": 0.02, "min_torsional_patch_radius_m": 0.01},
    "px_tor1_m6": {"torsional_patch_radius_m": 0.01, "min_torsional_patch_radius_m": 0.005, "mass_kg": 0.006},
}
VARIANTS = {
    "px_iter": {"solver_position_iterations": 32, "solver_velocity_iterations": 4},
    "px_mat": {"static_friction": 0.95, "dynamic_friction": 0.95, "friction_combine_mode": "max"},
    "px_off": {"contact_offset_m": 0.002, "rest_offset_m": 0.0},
    "px_all": {"solver_position_iterations": 32, "solver_velocity_iterations": 4, "static_friction": 0.95, "dynamic_friction": 0.95,
               "friction_combine_mode": "max", "contact_offset_m": 0.002, "rest_offset_m": 0.0, "mass_kg": 0.006},
}


def main() -> int:
    import sys
    OUT.mkdir(exist_ok=True)
    table = VARIANTS_TORSION if "--torsion" in sys.argv else VARIANTS
    for name, px in table.items():
        for src, dst in (("libero_spatial_environment.yaml", f"libero_spatial_environment_{name}.yaml"),
                         ("libero_spatial_environment_t5.yaml", f"libero_spatial_environment_{name}_t5.yaml")):
            p = BR / src
            if not p.exists():
                print("없음, 건너뜀:", src); continue
            cfg = yaml.safe_load(p.read_text())
            for b in BOWLS:
                if b in (cfg.get("objects") or {}):
                    cfg["objects"][b]["physx"] = dict(px)
            if "friction_combine_mode" in px:
                cfg.setdefault("robot", {}).setdefault("gripper_contact_material", {})["friction_combine_mode"] = px["friction_combine_mode"]
            (OUT / dst).write_text("# make_physx_variants.py 파생 (원본: %s) — 파지 시험 · 물리 설정 비교용\n" % src
                                   + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
            print("만듦:", OUT / dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
