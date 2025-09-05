# tally_boda_integration.py

import torch
import numpy as np
from torch.utils.data import Dataset

class TallySampler:
    """TALLY의 메인 학습 단계에서 사용할 데이터 샘플러입니다."""
    def __init__(self, dataset, batch_size, iter_num) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.iter_num = iter_num

    def __iter__(self):
        for i in range(self.iter_num):
            yield from self.dataset.get_balance_pair(self.batch_size)

    def __len__(self) -> int:
        return self.iter_num * 2 * self.batch_size


class TallyDatasetWrapper(Dataset):
    """
    BoDA 프레임워크의 표준 데이터셋을 TALLY 알고리즘이 사용할 수 있도록 변환하는 Wrapper 클래스입니다.
    """
    def __init__(self, boda_dataset, hparams):
        super().__init__()
        
        print("Initializing TallyDatasetWrapper...")
        self.underlying_dataset = boda_dataset
        self.hparams = hparams
        
        # 1. BoDA 데이터셋에서 모든 라벨과 도메인 정보를 추출합니다.
        self.labels = []
        self.domains = []
        self.domain_map = [] # 전역 인덱스 -> (도메인 인덱스, 해당 도메인 내 로컬 인덱스)
        
        for domain_idx, domain_data in enumerate(self.underlying_dataset.datasets):
            # SplitImageFolder의 targets 속성 사용
            domain_labels = domain_data.targets.tolist()
            self.labels.extend(domain_labels)
            self.domains.extend([domain_idx] * len(domain_labels))
            
            for local_idx in range(len(domain_labels)):
                self.domain_map.append((domain_idx, local_idx))

        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.domains = torch.tensor(self.domains, dtype=torch.long)
        
        print(f"Wrapper created. Total samples: {len(self.labels)}, Num domains: {len(self.underlying_dataset.datasets)}")

        # 2. TALLY가 사용하는 모든 통계 및 인덱스 정보를 초기화합니다.
        # (기존 TALLY의 dataset.py 코드를 대부분 그대로 가져옴)
        self.class_num = self.underlying_dataset.num_classes
        self.pairs = self.domains * self.class_num + self.labels

        self.unique_domains, self.domains_inverse = torch.unique(self.domains, return_inverse=True)
        self.unique_labels, self.labels_inverse = torch.unique(self.labels, return_inverse=True)
        self.unique_pairs = torch.unique(self.pairs)

        self.num_domains = len(self.unique_domains)
        self.num_classes = len(self.unique_labels)

        self.labels_indices = [torch.nonzero(self.labels == cls).squeeze(-1) for cls in range(self.num_classes)]
        self.domains_indices = [torch.nonzero(self.domains == loc).squeeze(-1) for loc in range(self.num_domains)]
        
        # hparams에서 통계 텐서의 차원 정보를 가져와야 함 (예: c, h, w)
        c, h, w = hparams.get('tally_c', 512), hparams.get('tally_h', 28), hparams.get('tally_w', 28)
        self.classes_norm = torch.randn(self.num_classes, c, h, w)
        self.domains_sigma = torch.randn(self.num_domains, c, 1, 1)
        self.domains_mean = torch.randn(self.num_domains, c, 1, 1)

    def get_balance_pair(self, batch_size):
        """TallySampler가 호출할 샘플링 메소드."""
        idx1, idx2 = [], []

        for _ in range(batch_size):
            class_idx = np.random.choice(self.num_classes, 1)[0]
            # 해당 클래스의 샘플이 없는 경우를 대비
            if len(self.labels_indices[class_idx]) == 0: continue
            feat_idx = np.random.choice(len(self.labels_indices[class_idx]), 1)[0]
            idx1.append(self.labels_indices[class_idx][feat_idx])

            domain_idx = np.random.choice(self.num_domains, 1)[0]
            if len(self.domains_indices[domain_idx]) == 0: continue
            feat_idx = np.random.choice(len(self.domains_indices[domain_idx]), 1)[0]
            idx2.append(self.domains_indices[domain_idx][feat_idx])

        return idx1 + idx2

    def update_statistics(self, norm, sig, mu, idx1, idx2):
        """TALLYAlgorithm이 호출할 통계 업데이트 메소드."""

        # GPU에 있는 인덱스 텐서를 CPU로 이동시킵니다.
        idx1 = idx1.cpu()
        idx2 = idx2.cpu()
        
        y_unique = self.labels_inverse[idx1].unique()
        d_unique = self.domains_inverse[idx2].unique()

        classes_feat_update = [norm[self.labels_inverse[idx1] == cls].mean(dim=0, keepdim=True) for cls in y_unique]
        classes_feat_update = torch.cat(classes_feat_update)

        domains_mean_update = [mu[self.domains_inverse[idx2] == loc].mean(dim=0, keepdim=True) for loc in d_unique]
        domains_mean_update = torch.cat(domains_mean_update)

        domains_sigma_update = [sig[self.domains_inverse[idx2] == loc].mean(dim=0, keepdim=True) for loc in d_unique]
        domains_sigma_update = torch.cat(domains_sigma_update)
        
        # EMA 업데이트
        self.classes_norm[y_unique] = 0.8 * self.classes_norm[y_unique] + 0.2 * classes_feat_update
        self.domains_mean[d_unique] = 0.8 * self.domains_mean[d_unique] + 0.2 * domains_mean_update
        self.domains_sigma[d_unique] = 0.8 * self.domains_sigma[d_unique] + 0.2 * domains_sigma_update

    def __getitem__(self, idx):
        """
        전역 인덱스(idx)를 받아 원본 BoDA 데이터셋에서 (이미지, 라벨)을 가져온 후,
        Wrapper가 관리하는 통계 정보를 추가하여 TALLY가 요구하는 최종 출력 형태를 만듭니다.
        """
        # 전역 인덱스를 (도메인, 로컬 인덱스)로 변환
        domain_idx, local_idx = self.domain_map[idx]
        
        # 원본 데이터셋에서 이미지와 라벨 가져오기
        img, label = self.underlying_dataset.datasets[domain_idx][local_idx]
        
        # Wrapper가 관리하는 통계 정보 가져오기
        norm = self.classes_norm[self.labels_inverse[idx]]
        sigma = self.domains_sigma[self.domains_inverse[idx]]
        mean = self.domains_mean[self.domains_inverse[idx]]
        domain = self.domains[idx]
        
        # TALLY가 요구하는 최종 튜플 반환
        return img, label, domain, idx, norm, sigma, mean

    def __len__(self):
        return len(self.labels)