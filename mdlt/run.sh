#!/bin/bash

#SBATCH --job-name=sweep_TALLY_PACS
#SBATCH --partition=laal_3090
#SBATCH --nodes=1    
#SBATCH --gres=gpu:0
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=6
#SBATCH --output=./slurm_logs/S-%x.%j.out     

# cd /home/hyunggyu/imbalance/multi-domain-imbalance

# nvidia-smi

source ~/.bashrc
source ~/anaconda3/bin/activate
conda activate spec

#     "VLCS",
#     "PACS",
#     "OfficeHome",
#     "TerraIncognita",
#     "DomainNet"

python -m mdlt.train \
  --dataset TerraIncognita \
  --algorithm MLIR \
  --output_folder_name res50_test \
  --data_dir /home/shared \
  --output_dir ./output \
  --hparams '{"resnet50": true}' \
  --hparams_seed 0 \
  --seed 0


# python -m mdlt.scripts.download --data_dir /home/shared

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

# python -m mdlt.scripts.collect_results --input_dir ./output/sweep_PACS_BoDA_mahal_1stage_mulSeed

# python -m mdlt.train \
#   --algorithm ERM \
#   --output_folder_name DomainNet126_ERM \
#   --data_dir /home/shared \
#   --output_dir ./output \
#   --hparams_seed 0 \
#   --seed 0 \
#   --dataset DomainNet126
# --stage1_folder ./sweep_PACS_BoDA_mahal_1stage_mulSeed \
# --stage1_algo 'BoDA' \
# --boda_dist_measure "mahalanobis" \

# MDLD_TIME1
python -m mdlt.sweep launch \
  --output_folder_name  sweep_TALLY_PACS \
  --algorithms 'TALLY' \
  --data_dir /home/shared \
  --dataset PACS \
  --output_dir ./output \
  --skip_confirmation \

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
#   --input_dir /home/hyunggyu/imbalance/multi-domain-imbalance/output/sweep_res18_mydataset2
  


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