#!/bin/bash

#SBATCH --job-name=MDLT_KL_PACS
#SBATCH --partition=laal_a6000
#SBATCH --nodes=1    
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=5
#SBATCH --output=./slurm_logs_time/S-%x.%j.out     

cd /home/hyunggyu/imbalance/multi-domain-imbalance

# nvidia-smi

# python -m mdlt.train \
#   --dataset PACS \
#   --algorithm IRM \
#   --output_folder_name res18_test \
#   --data_dir /home/shared \
#   --output_dir ./output \
#   --hparams '{"resnet18": true}'


# python -m mdlt.scripts.download --data_dir /home/shared

# sweep 실행
# python -m mdlt.sweep launch \
#   --output_folder_name sweep_timecheck_res18 \
#   --data_dir /home/shared \
#   --output_dir ./output \
#   --hparams '{"resnet18": true}' \
#   --skip_confirmation


# INCOMPLETE 디렉토리 삭제하고 재실행
# Step 1: INCOMPLETE된 실험 디렉토리 삭제
python -m mdlt.sweep delete_incomplete \
  --output_folder_name sweep_KL \
  --data_dir /home/shared \
  --output_dir ./output \
  --hparams '{"resnet18": true}' \
  --skip_confirmation

# ALG별 SWEEP 실행
python -m mdlt.sweep launch \
  --output_folder_name sweep_KL \
  --algorithms 'KL' \
  --data_dir /home/shared \
  --dataset PACS \
  --output_dir ./output \
  --hparams '{"resnet18": true}' \
  --n_hparams 1 \
  --skip_confirmation

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