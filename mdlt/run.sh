#!/bin/bash

#SBATCH --job-name=BoDA_hparam_DN126
#SBATCH --partition=laal_3090
#SBATCH --nodes=1    
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=6
#SBATCH --output=./slurm_logs/S-%x.%j.out     

cd /home/hyunggyu/imbalance/multi-domain-imbalance

# sweep 실행
# python -m mdlt.sweep launch \
#   --output_folder_name sweep_timecheck_res18 \
#   --data_dir /home/shared \
#   --output_dir ./output \
#   --hparams '{"resnet18": true}' \
#   --skip_confirmation


# # INCOMPLETE 디렉토리 삭제하고 재실행
# # Step 1: INCOMPLETE된 실험 디렉토리 삭제
# python -m mdlt.sweep delete_incomplete \
#   --output_folder_name sweep_KL \
#   --data_dir /home/shared \
#   --output_dir ./output \
#   --hparams '{"resnet18": true}' \
#   --skip_confirmation

# ALG별 SWEEP 실행
# python -m mdlt.train \
#   --output_folder_name all_mahal_BoDA_1stage \
#   --algorithm 'BoDA' \
#   --data_dir /home/shared \
#   --output_dir ./output \
#   --boda_dist_measure "mahalanobis" \

# python -m mdlt.train \
#   --algorithm MLIR \
#   --output_folder_name DN126_MLIR \
#   --data_dir /home/shared \
#   --output_dir ./output \
#   --hparams_seed 0 \
#   --seed 0 \
#   --dataset DomainNet126
# --stage1_folder ./sweep_PACS_BoDA_mahal_1stage_mulSeed \
# --stage1_algo 'BoDA' \
# --boda_dist_measure "mahalanobis" \

# MDLD_TIME1
# python -m mdlt.sweep launch \
#   --output_folder_name  sweep_DN126_MLIR \
#   --algorithms MLIR \
#   --data_dir /home/shared \
#   --dataset DomainNet126 \
#   --output_dir ./output \
#   --skip_confirmation \
#   --n_hparams 1 \
#   --n_trials 3  # number of random seeds

# # 단일 ALG, DATASET, HPARAMS_SEED 고정, 여러 SEED 실행
# ALGO="MMD"
# OUTPUT_FOLDER_NAME="DN126_MMD"
# DATA_DIR="/home/shared"
# OUTPUT_DIR="./output"
# HPARAMS_SEED=0
# DATASET="DomainNet126"

# # 시작값과 개수 설정 (원하는 만큼 반복)
# START=45
# COUNT=20   # 몇 개의 시드를 돌릴지 지정 (예: 5,6,7 → COUNT=3)

# # 기본 시드 목록을 seq로 생성
# SEEDS=($(seq $START $((START + COUNT - 1))))

# # 인자를 주면 그 시드들로 덮어씀
# if [ "$#" -gt 0 ]; then
#   SEEDS=("$@")
# fi

# echo "[INFO] Running seeds: ${SEEDS[*]}"
# for SEED in "${SEEDS[@]}"; do
#   echo "==> Launching seed ${SEED}"
#   python -m mdlt.train \
#     --algorithm "${ALGO}" \
#     --output_folder_name "${OUTPUT_FOLDER_NAME}" \
#     --data_dir "${DATA_DIR}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --hparams_seed "${HPARAMS_SEED}" \
#     --seed "${SEED}" \
#     --dataset "${DATASET}"
# done

# echo "[DONE] All runs finished."


# MDLD_TIME1
# python -m mdlt.sweep launch \
#   --output_folder_name sweep_timecheck_res18_DomainNet \
#   --algorithms 'ERM' 'IRM' 'GroupDRO' 'Mixup' 'MLDG' 'CORAL' 'MMD' \
#   --data_dir /home/shared \
#   --dataset DomainNet \
#   --output_dir ./output \
#   --hparams '{"resnet18": true}' \
#   --n_hparams 1 \
#   --skip_confirmation

# MDLD_TIME2
# python -m mdlt.sweep launch \
#   --output_folder_name sweep_timecheck_res18_DomainNet \
#   --data_dir /home/shared \
#   --dataset DomainNet \
#   --algorithms 'DANN' 'CDANN' 'MTL' 'SagNet' 'Fish' 'ReSamp' 'ReWeight' \
#   --output_dir ./output \
#   --hparams '{"resnet18": true}' \
#   --n_hparams 1 \
#   --skip_confirmation

# MDLD_TIME3
  # python -m mdlt.sweep launch \
  #   --output_folder_name sweep_timecheck_res18_DomainNet \
  #   --algorithms 'SqrtReWeight' 'CBLoss' 'Focal' 'LDAM' 'BSoftmax' 'BoDA' \
  #   --data_dir /home/shared \
  #   --dataset DomainNet \
  #   --output_dir ./output \
  #   --hparams '{"resnet18": true}' \
  #   --n_hparams 1 \
  #   --skip_confirmation

# collect 실행
# python -m mdlt.scripts.collect_results \
#   --input_dir /home/hyunggyu/imbalance/multi-domain-imbalance/output/sweep_DN126_SqrtRW-BoDA
  


# INCOMPLETE 디렉토리 삭제하고 재실행
# Step 1: INCOMPLETE된 실험 디렉토리 삭제
# python -m mdlt.sweep delete_incomplete \
#   --output_folder_name sweep_res18_mydataset2 \
#   --data_dir /home/shared \
#   --output_dir ./output \
#   --hparams '{"resnet18": true}' \
#   --skip_confirmation

  # --algorithms 'BODA' \
# Step 2: 다시 launch




# ALGORITHMS list
# --algorithms 'ERM' \
# --algorithms 'IRM' \
# --algorithms 'GroupDRO' \
# --algorithms 'Mixup' \
# --algorithms 'MLDG' \
# --algorithms 'CORAL' \
# --algorithms 'MMD' \
# --algorithms 'DANN' \
# --algorithms 'CDANN' \
# --algorithms 'MTL' \
# --algorithms 'SagNet' \
# --algorithms 'Fish' \
# --algorithms 'ReSamp' \
# --algorithms 'ReWeight' \
# --algorithms 'SqrtReWeight' \
# --algorithms 'CBLoss' \
# --algorithms 'Focal' \
# --algorithms 'LDAM' \
# --algorithms 'BSoftmax' \
# --algorithms 'CRT' \
# --algorithms 'BoDA' \
# --algorithms 'KL' \

# DATASETS = [
#     # Debug
#     "Debug28",
#     "Debug224",
#     # Small MDLT datasets
#     "ImbalancedColoredMNIST",
#     "ImbalancedRotatedMNIST",
#     "ImbalancedDigits",
#     # Big MDLT datasets
#     "VLCS",
#     "PACS",
#     "OfficeHome",
#     "TerraIncognita",
#     "DomainNet"
# ]