#!/usr/bin/env bash
# v4b 학습셋으로 OpenVLA 미세조정 — 빌드 → npz 대조 → GPU 1 확인 → 학습.  서버(sch@10.115.10.87)에서 실행.
#   setsid nohup bash ~/rlds_builder_v4b/train_v4b.sh > /dev/null 2>&1 &
# 기록: ~/openvla_runs/logs/v4b_<날짜시각>.log
# 서버의 openvla 저장소 코드와 9 월 데이터셋(~/tensorflow_datasets/isaac_libero_spatial)은 건드리지 않는다.
set -uo pipefail
source ~/anaconda3/etc/profile.d/conda.sh && conda activate openvla
TAG=v4b_$(date +%m%d_%H%M)
mkdir -p ~/openvla_runs/logs
LOG=~/openvla_runs/logs/$TAG.log
exec > >(tee -a "$LOG") 2>&1
echo "== $TAG 시작 $(date)"

BASE=${BASE:-openvla/openvla-7b-finetuned-libero-spatial}   # 출발점: 공개 LIBERO-Spatial 체크포인트 (제로샷과 같은 모델)
DATA_DIR=~/tensorflow_datasets_v4b
NPZ=~/datasets/finetune_dataset_v4b

echo "[1/4] RLDS 빌드 → $DATA_DIR"
if [ -d "$DATA_DIR/isaac_libero_spatial/1.0.0" ]; then
  echo "  이미 빌드됨, 건너뜀"
else
  cd ~/rlds_builder_v4b/isaac_libero_spatial || exit 1
  ISAAC_LIBERO_NPZ_ROOT=$NPZ CUDA_VISIBLE_DEVICES="" tfds build --data_dir "$DATA_DIR" || { echo "빌드 실패 — 중단"; exit 1; }
fi

echo "[2/4] RLDS ↔ npz 대조"
cd ~/rlds_builder_v4b || exit 1
CUDA_VISIBLE_DEVICES="" python check_rlds_vs_npz.py --data-dir "$DATA_DIR" --npz-root "$NPZ" | tee /tmp/check_$TAG.txt
grep -q "판정: 문제 없음" /tmp/check_$TAG.txt || { echo "대조 실패 — 학습하지 않음"; exit 1; }

echo "[3/4] GPU 1 확인 (GPU 0 은 다른 사용자 작업이 있어 쓰지 않음)"
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
echo "  GPU 1 사용 중 메모리 ${USED} MiB"
[ "${USED:-99999}" -lt 2000 ] || { echo "GPU 1 이 비어 있지 않음 — 중단"; exit 1; }
ls -d ~/.cache/huggingface/hub/models--openvla--openvla-7b-finetuned-libero-spatial 2>/dev/null && echo "  출발 체크포인트: 캐시에 있음" || echo "  출발 체크포인트: 캐시에 없음 → 내려받음"

echo "[4/4] 학습 (LoRA r32 · 배치 16 · lr 5e-5 · 8000 스텝 · 2000 스텝마다 저장 · 이미지 증강)"
cd ~/openvla || exit 1
python vla-scripts/finetune.py --help 2>/dev/null | head -60
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "$BASE" \
  --data_root_dir "$DATA_DIR" \
  --dataset_name isaac_libero_spatial \
  --run_root_dir ~/openvla_runs \
  --adapter_tmp_dir ~/openvla_runs/adapter_tmp_$TAG \
  --run_id_note "$TAG" \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --image_aug True \
  --max_steps 8000 \
  --save_steps 2000 \
  --save_latest_checkpoint_only False
echo "== $TAG 끝 $(date)  종료 코드 $?"
