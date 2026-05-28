#!/bin/bash
set -e

RUN_NAME="${RUN_NAME:-SVR2026}"

echo "=== Iniciando Re-Texture am ${RUN_NAME} =="
python extract_texture.py configs/texture_gaussian3d.yaml --resume_from output/${RUN_NAME}/texture_gaussian3d/checkpoints/40000.pth --save_path output/${RUN_NAME}/localized_custom_gs/texture_original.png
python lc_analyzer.py
echo "=== Re-Texture finalizado com sucesso! ==="