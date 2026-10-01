#!/usr/bin/env python3
"""원본 환경 코드는 그대로 두고, 물체별 물리 설정을 yaml 로 바꿀 수 있는 사본을 만든다.

    python3 make_env_physx_copy.py
    → ~/isaac_libero/bridge/libero_spatial_env_isaac601_physx.py

yaml 에서 쓸 수 있는 키 (없으면 원본과 같은 동작)
  objects.<key>.physx:
      mass_kg: 0.006                      # 질량 고정
      solver_position_iterations: 32      # 강체 반복 계산 횟수 (PhysX 기본 4)
      solver_velocity_iterations: 4
      contact_offset_m: 0.002             # 접촉 계산 시작 거리
      rest_offset_m: 0.0                  # 물체가 실제로 멈추는 간격
      static_friction: 0.95               # 물체 물리 재질 (지정하면 새 재질을 만들어 충돌 prim 에 붙임)
      dynamic_friction: 0.95
      friction_combine_mode: max          # average | min | multiply | max
      torsional_patch_radius_m: 0.01      # 비틀림 마찰 반경 (PhysX 기본 0 = 꺼짐)
      min_torsional_patch_radius_m: 0.005 # 접촉이 작아도 쓰는 최소 비틀림 반경
  robot.gripper_contact_material.friction_combine_mode: max
"""

import shutil
from pathlib import Path

SRC = Path.home() / "isaac_libero/bridge/libero_spatial_env_isaac601.py"
DST = Path.home() / "isaac_libero/bridge/libero_spatial_env_isaac601_physx.py"

OBJ_ANCHOR = '''            print(f"[Object] {key}: {usd_path} -> {root_path}; rigid={rigid_path}")\n'''
OBJ_ADD = '''            print(f"[Object] {key}: {usd_path} -> {root_path}; rigid={rigid_path}")
            px = cfg.get("physx") or {}
            if px:
                from pxr import PhysxSchema as _Px
                rigid_prim = self.stage.GetPrimAtPath(rigid_path)
                done = []
                if px.get("mass_kg") is not None:
                    UsdPhysics.MassAPI.Apply(rigid_prim).CreateMassAttr().Set(float(px["mass_kg"])); done.append("mass=%.4f" % float(px["mass_kg"]))
                if px.get("solver_position_iterations") is not None or px.get("solver_velocity_iterations") is not None:
                    rb = _Px.PhysxRigidBodyAPI.Apply(rigid_prim)
                    if px.get("solver_position_iterations") is not None:
                        rb.CreateSolverPositionIterationCountAttr().Set(int(px["solver_position_iterations"])); done.append("pos_iter=%d" % int(px["solver_position_iterations"]))
                    if px.get("solver_velocity_iterations") is not None:
                        rb.CreateSolverVelocityIterationCountAttr().Set(int(px["solver_velocity_iterations"])); done.append("vel_iter=%d" % int(px["solver_velocity_iterations"]))
                coll = [p for p in Usd.PrimRange(self.stage.GetPrimAtPath(root_path)) if p.HasAPI(UsdPhysics.CollisionAPI)]
                if px.get("contact_offset_m") is not None or px.get("rest_offset_m") is not None:
                    for p in coll:
                        ca = _Px.PhysxCollisionAPI.Apply(p)
                        if px.get("contact_offset_m") is not None:
                            ca.CreateContactOffsetAttr().Set(float(px["contact_offset_m"]))
                        if px.get("rest_offset_m") is not None:
                            ca.CreateRestOffsetAttr().Set(float(px["rest_offset_m"]))
                    done.append("offsets(contact=%s, rest=%s) on %d colliders" % (px.get("contact_offset_m"), px.get("rest_offset_m"), len(coll)))
                if px.get("torsional_patch_radius_m") is not None or px.get("min_torsional_patch_radius_m") is not None:
                    for p in coll:
                        ca = _Px.PhysxCollisionAPI.Apply(p)
                        if px.get("torsional_patch_radius_m") is not None:
                            ca.CreateTorsionalPatchRadiusAttr().Set(float(px["torsional_patch_radius_m"]))
                        if px.get("min_torsional_patch_radius_m") is not None:
                            ca.CreateMinTorsionalPatchRadiusAttr().Set(float(px["min_torsional_patch_radius_m"]))
                    done.append("torsional(radius=%s, min=%s) on %d colliders" % (px.get("torsional_patch_radius_m"), px.get("min_torsional_patch_radius_m"), len(coll)))
                if px.get("static_friction") is not None:
                    mat_path = "/World/Materials/ObjPhysx_%s" % key
                    mat = UsdShade.Material.Define(self.stage, mat_path)
                    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                    pm.CreateStaticFrictionAttr().Set(float(px["static_friction"]))
                    pm.CreateDynamicFrictionAttr().Set(float(px.get("dynamic_friction", px["static_friction"])))
                    pm.CreateRestitutionAttr().Set(0.0)
                    if px.get("friction_combine_mode"):
                        _Px.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr().Set(str(px["friction_combine_mode"]))
                    for p in coll:
                        UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
                    done.append("material(static=%.2f, dynamic=%.2f, combine=%s) on %d colliders" % (
                        float(px["static_friction"]), float(px.get("dynamic_friction", px["static_friction"])), px.get("friction_combine_mode", "default"), len(coll)))
                print(f"[Object Physx] {key}: " + "; ".join(done))
'''

MAT_ANCHOR = '''        physics_material.CreateRestitutionAttr(restitution)
        self.gripper_contact_material_path = material_path
'''
MAT_ADD = '''        physics_material.CreateRestitutionAttr(restitution)
        if cfg.get("friction_combine_mode"):
            from pxr import PhysxSchema as _Px
            _Px.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr().Set(str(cfg["friction_combine_mode"]))
            print("[Gripper Contact Material] friction_combine_mode=%s" % cfg["friction_combine_mode"])
        self.gripper_contact_material_path = material_path
'''


def main() -> int:
    s = SRC.read_text()
    for name, anchor in (("물체 생성", OBJ_ANCHOR), ("손가락 재질", MAT_ANCHOR)):
        if s.count(anchor) != 1:
            print("기준 줄을 찾지 못함: %s (개수 %d) — 원본이 바뀌었는지 확인" % (name, s.count(anchor))); return 1
    s = s.replace(OBJ_ANCHOR, OBJ_ADD, 1).replace(MAT_ANCHOR, MAT_ADD, 1)
    if DST.exists():
        shutil.copy2(DST, DST.with_suffix(".py.prev"))
    DST.write_text(s)
    print("만듦:", DST, "(원본은 그대로)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
