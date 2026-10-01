#!/usr/bin/env bash
# 학습 서버 확인 (읽기만 함).  실행: bash -li ~/rlds_builder_v4b/inspect_server.sh
echo "== 빌더 파일"; ls ~/rlds_builder_v4b ~/rlds_builder_v4b/isaac_libero_spatial
echo "== 데이터"; find ~/datasets/finetune_dataset_v4b -name "*.npz" | wc -l; du -sh ~/datasets/finetune_dataset_v4b
echo "== 등록 내용 (configs.py)"; grep -n -A 14 '"isaac_libero_spatial"' ~/openvla/prismatic/vla/datasets/rlds/oxe/configs.py | head -18
echo "== 9 월 빌드"; ls ~/tensorflow_datasets/isaac_libero_spatial/ 2>/dev/null
echo "== 파이썬 환경 (torch · tensorflow_datasets 가 모두 있는 것)"
command -v conda >/dev/null || for c in ~/miniconda3 ~/anaconda3 ~/miniforge3; do [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh"; done
conda env list 2>/dev/null | grep -v "^#"
for p in $(conda env list 2>/dev/null | awk '!/^#/ && NF {print $NF}'); do
  [ -x "$p/bin/python" ] && "$p/bin/python" -c "import torch, tensorflow_datasets as t; print('   후보', '$p', 'torch', torch.__version__, 'cuda', torch.version.cuda, 'tfds', t.__version__)" 2>/dev/null
done
echo "== 9 월 학습 폴더"; for r in $(ls -t ~/openvla_runs | head -3); do echo "-- $r"; ls ~/openvla_runs/$r | head -6; done
echo "== 학습 스크립트"; ls ~/openvla/vla-scripts/ 2>/dev/null | head; ls ~/*.sh ~/openvla/*.sh ~/openvla_runs/*.sh 2>/dev/null | head
