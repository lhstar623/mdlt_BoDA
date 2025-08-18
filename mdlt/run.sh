#!/bin/bash

#SBATCH --job-name=MDLT_KL_PACS
#SBATCH --partition=laal_a6000
#SBATCH --nodes=1    
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=5
#SBATCH --output=./slurm_logs_time/S-%x.%j.out     

cd /home/hyunggyu/imbalance/multi-domain-imbalance

#!/bin/bash
#SBATCH --job-name=CAWRA_TAROT_run_BoDA
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --partition=laal_3090
#SBATCH --time=UNLIMITED
# --- :톱니바퀴: 설정: 여기에 실행할 데이터셋 목록을 추가하세요 ---
# "DomainNet" "OfficeHome" "TerraIncognita" "VLCS" "PACS"
DATASETS=("PACS" "VLCS" "TerraIncognita" "OfficeHome" "DomainNet")
# --- 공통 경로 설정 ---
DATA_DIR="/home/shared"
BASE_OUTPUT_DIR="./output"
# --- 반복문 시작 ---
for dataset in "${DATASETS[@]}"; do
    echo "================================================="
    echo ":로켓: Starting CAWRA_TAROT training for dataset: ${dataset}"
    echo "================================================="
    # :흰색_확인_표시: 각 데이터셋에 대한 파이썬 스크립트 실행
    #    --algorithm을 CAWRA_TAROT으로 변경하고 관련 인자들을 추가합니다.
    python -m mdlt.train \
      --dataset "${dataset}" \
      --algorithm CAWRA_TAROT \
      --output_folder_name "CAWRA_TAROT_${dataset}" \
      --data_dir "${DATA_DIR}" \
      --output_dir "${BASE_OUTPUT_DIR}" \
      --seed 0 \
      --use_boda True \
      --use_calibration True \
      --use_xent True \
      --boda_dist_measure "mahalanobis" \
      --macro_weight 1.0 \
      --target_adv_weight 1.0 \
      --pgd_eps 8.0
    echo ":흰색_확인_표시: Finished training for dataset: ${dataset}"
    echo "-------------------------------------------------"
    echo ""
done
echo ":짠: All dataset runs are complete."


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


# # INCOMPLETE 디렉토리 삭제하고 재실행
# # Step 1: INCOMPLETE된 실험 디렉토리 삭제
# python -m mdlt.sweep delete_incomplete \
#   --output_folder_name sweep_KL \
#   --data_dir /home/shared \
#   --output_dir ./output \
#   --hparams '{"resnet18": true}' \
#   --skip_confirmation

# # ALG별 SWEEP 실행
# python -m mdlt.sweep launch \
#   --output_folder_name sweep_KL \
#   --algorithms 'KL' \
#   --data_dir /home/shared \
#   --dataset PACS \
#   --output_dir ./output \
#   --hparams '{"resnet18": true}' \
#   --n_hparams 1 \
#   --skip_confirmation

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