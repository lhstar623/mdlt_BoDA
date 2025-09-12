#!/bin/bash

#SBATCH --job-name=DN_zip
#SBATCH --partition=laal_3090
#SBATCH --nodes=1    
#SBATCH --gres=gpu:0
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=4
#SBATCH --output=./slurm_logs/S-%x.%j.out     

# 이동할 프로젝트 경로
cd /home/hyunggyu/imbalance/multi-domain-imbalance/mdlt

# 데이터셋을 저장할 경로
DATA_DIR="/home/hyunggyu/imbalance/multi-domain-imbalance/mdlt/dataset"
# mkdir -p $DATA_DIR

# 다운로드할 URL 리스트
URLS=(
    "https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation/blob/master/data/domainnet-126/clipart_list.txt"
    "https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation/blob/master/data/domainnet-126/painting_list.txt"
    "https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation/blob/master/data/domainnet-126/real_list.txt"
    "https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation/blob/master/data/domainnet-126/sketch_list.txt"
)

echo "==== 데이터셋 다운로드 시작 ===="

for url in "${URLS[@]}"; do
    echo "Downloading $url ..."
    wget -c -P $DATA_DIR $url
done

echo "==== 데이터셋 다운로드 완료 ===="