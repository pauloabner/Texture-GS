#!/bin/bash
set -e

RUN_NAME="${RUN_NAME:-SVR2026}"

echo "=== Iniciando os treinos com RUN_NAME: ${RUN_NAME} =="

python train.py configs/gaussian3d_base.yaml --run_name "${RUN_NAME}"
python extract_pcd.py configs/gaussian3d_base.yaml --resume_from "output/${RUN_NAME}/gaussian3d_base/checkpoints/30000.pth" --save_path "output/${RUN_NAME}/gaussian3d_base/pcd.npy"
python train.py configs/uv_map.yaml --run_name "${RUN_NAME}"
python train.py configs/texture_gaussian3d.yaml --run_name "${RUN_NAME}"

echo "=== Treinos finalizados com sucesso! ==="