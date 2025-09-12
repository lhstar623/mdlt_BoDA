import argparse
import collections
import json
import os
import random
import sys
import time
import shutil
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from tensorboard_logger import Logger
from tqdm import tqdm

from mdlt import hparams_registry
from mdlt.dataset import datasets
from mdlt.learning import algorithms
from mdlt.utils import misc
from mdlt.dataset.fast_dataloader import InfiniteDataLoader, FastDataLoader

from mdlt.dataset.tally_boda_integration import TallyDatasetWrapper, TallySampler
from torch.utils.data import DataLoader

# hyunggyu - for time &  c/gpu usage logging
# import pynvml
# import psutil
# import threading
# import time

# ──────────────────────────────────────────────────────────────────────────────
# SubsetWithTargets: torch.utils.data.Subset 에 .targets 속성까지 복사
# ──────────────────────────────────────────────────────────────────────────────
class SubsetWithTargets(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        if hasattr(dataset, "targets"):            # TensorDataset, SplitImageFolder 등
            self.targets = np.asarray(dataset.targets)[indices]
        elif hasattr(dataset, "tensors"):          # TensorDataset
            self.targets = dataset.tensors[1].numpy()[indices]
        else:
            raise AttributeError(
                "custom_counts: 대상 데이터셋에 targets 속성을 찾을 수 없습니다."
            )

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


# class ResourceTracker:
#     def __init__(self, device_index=0, interval=1):
#         self.device_index = device_index
#         self.interval = interval
#         self._stop_event = threading.Event()

#         # GPU 통계
#         self.gpu_min = float('inf')
#         self.gpu_max = float('-inf')
#         self.gpu_sum = 0
#         self.gpu_count = 0

#         self.mem_min = float('inf')
#         self.mem_max = float('-inf')
#         self.mem_sum = 0
#         self.mem_count = 0

#         # CPU 메모리 (RSS, MB)
#         self.cpu_min = float('inf')
#         self.cpu_max = float('-inf')
#         self.cpu_sum = 0
#         self.cpu_count = 0

#         self.start_time = None
#         self.end_time = None

#     def _track(self):
#         pynvml.nvmlInit()
#         handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
#         process = psutil.Process()

#         self.start_time = time.time()

#         while not self._stop_event.is_set():
#             # GPU
#             util = pynvml.nvmlDeviceGetUtilizationRates(handle)
#             mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
#             gpu = util.gpu
#             mem_used = mem.used / 1024 / 1024

#             self.gpu_min = min(self.gpu_min, gpu)
#             self.gpu_max = max(self.gpu_max, gpu)
#             self.gpu_sum += gpu
#             self.gpu_count += 1

#             self.mem_min = min(self.mem_min, mem_used)
#             self.mem_max = max(self.mem_max, mem_used)
#             self.mem_sum += mem_used
#             self.mem_count += 1

#             # CPU
#             rss_used = process.memory_info().rss / 1024 / 1024  # MB
#             self.cpu_min = min(self.cpu_min, rss_used)
#             self.cpu_max = max(self.cpu_max, rss_used)
#             self.cpu_sum += rss_used
#             self.cpu_count += 1

#             time.sleep(self.interval)

#         self.end_time = time.time()
#         pynvml.nvmlShutdown()

#     def start(self):
#         self.thread = threading.Thread(target=self._track)
#         self.thread.start()

#     def stop(self):
#         self._stop_event.set()
#         self.thread.join()

#     def save(self, path):
#         gpu_avg = self.gpu_sum / self.gpu_count if self.gpu_count else 0
#         mem_avg = self.mem_sum / self.mem_count if self.mem_count else 0
#         cpu_avg = self.cpu_sum / self.cpu_count if self.cpu_count else 0
#         duration = self.end_time - self.start_time if self.end_time and self.start_time else 0

#         with open(path, 'w') as f:
#             f.write(f"Tracking Duration (sec): {duration:.2f}\n")
#             f.write(f"GPU Utilization (%): min={self.gpu_min}, max={self.gpu_max}, avg={gpu_avg:.2f}\n")
#             f.write(f"GPU Memory Usage (MiB): min={self.mem_min:.0f}, max={self.mem_max:.0f}, avg={mem_avg:.2f}\n")
#             f.write(f"CPU Memory Usage (MiB): min={self.cpu_min:.0f}, max={self.cpu_max:.0f}, avg={cpu_avg:.2f}\n")



if __name__ == "__main__":
    
    
    # 1. 추적 시작
    # tracker = ResourceTracker(interval=5)
    # tracker.start()
    
    parser = argparse.ArgumentParser(description='Multi-Domain LT')
    # training
    parser.add_argument('--dataset', type=str, default="PACS", choices=datasets.DATASETS)
    parser.add_argument('--algorithm', type=str, default="ERM", choices=algorithms.ALGORITHMS)
    parser.add_argument('--output_folder_name', type=str, default='debug')
    # imbalance related
    parser.add_argument('--imb_type', type=str, default="eeee",
                        help='Length should be equal to # of envs, each refers to imb_type within that env')
    parser.add_argument('--imb_factor', type=float, default=0.1)

    # scuttie - for custom counts
    parser.add_argument('--custom_counts', type=str, default=None,
                        help='JSON list (len=#env) of per‑class counts for *train* split')
    parser.add_argument('--cross_env_gamma',
                        type=float,
                        default=None,
                        help='(BoDA 전용) cross_env_gamma 값을 수동으로 override')
    parser.add_argument(
        '--use_boda',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Whether to apply BoDA loss (True/False).'
    )
    parser.add_argument(
        '--use_xent',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Whether to include cross-entropy loss (True/False)'
    )
    parser.add_argument(
        '--use_calibration',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Whether to include calibration (True/False)'
    )
    parser.add_argument(
        '--boda_dist_measure',
        type=str,
        default='coral',
        choices=['coral', 'mahalanobis'],
        help='(BoDA only) Distance measure for macro-alignment penalty.'
    )
    parser.add_argument('--global_weight',
                        type=float,
                        default=0.0,
                        help='Weight β for the global imbalance loss term')
    parser.add_argument('--macro_weight',
                        type=float,
                        default=None,
                        help='(BoDA only) Weight for the macro-alignment penalty.')
    parser.add_argument('--target_adv_weight', type=float, default=None,
                        help='(CAWRA_TAROT only) Weight for the target adversarial loss.')
    parser.add_argument('--pgd_eps', type=float, default=None,
                        help='(CAWRA_TAROT only) Epsilon for PGD attack.')


    # others
    # parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--data_dir', type=str, default="/home/shared")

    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for random hparams (0 for "default hparams")')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--selected_envs', type=int, nargs='+', default=None, help='Train only on selected envs')
    # two-stage related
    parser.add_argument('--stage1_folder', type=str, default='vanilla')
    parser.add_argument('--stage1_algo', type=str, default='ERM')
    # checkpoints
    parser.add_argument('--resume', '-r', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--checkpoint_freq', type=int, default=None, help='Checkpoint every N steps')
    parser.add_argument('--skip_model_save', action='store_true')
    args = parser.parse_args()

    start_step = 0
    args.best_val_acc = 0
    best_env_acc = {}
    best_shot_acc = {}
    best_class_acc = collections.defaultdict(list)
    store_prefix = f"{args.dataset}_{args.imb_type}_{args.imb_factor}" if 'Imbalance' in args.dataset else args.dataset
    args.store_name = f"{store_prefix}_{args.algorithm}_hparams{args.hparams_seed}_seed{args.seed}"
    if args.selected_envs is not None:
        args.store_name = f"{args.store_name}_env{str(args.selected_envs).replace(' ', '')[1:-1]}"

    misc.prepare_folders(args)
    args.output_dir = os.path.join(args.output_dir, args.output_folder_name, args.store_name)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    tb_logger = Logger(logdir=args.output_dir, flush_secs=2)

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, misc.seed_hash(args.hparams_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    if 'Imbalance' in args.dataset:
        hparams.update({'imb_type_per_env': [misc.IMBALANCE_TYPE[x] for x in args.imb_type],
                        'imb_factor': args.imb_factor})

    # BoDA 알고리즘일 때, CLI로 넘긴 gamma 값으로 덮어쓰기
    if 'BoDA' in args.algorithm or args.algorithm == 'CAWRA_TAROT':
        if args.cross_env_gamma is not None:
            hparams['cross_env_gamma'] = args.cross_env_gamma
        hparams['use_boda'] = args.use_boda
        hparams['use_xent'] = args.use_xent
        hparams['global_weight'] = args.global_weight
        hparams['use_calibration'] = args.use_calibration
        hparams['boda_dist_measure'] = args.boda_dist_measure
        if args.macro_weight is not None:
            hparams['macro_weight'] = args.macro_weight

    if args.algorithm == 'CAWRA_TAROT':
        if args.target_adv_weight is not None:
            hparams['target_adv_weight'] = args.target_adv_weight
        if args.pgd_eps is not None:
            hparams['pgd_eps'] = args.pgd_eps / 255.0 # 입력은 8, 16 등으로 받고 255로 나눠줌

    # 전달 받은 custom_counts → hparams 로 넘겨 데이터셋 생성 단계에서 사용
    if args.custom_counts:
        hparams['custom_counts'] = json.loads(args.custom_counts)
    else:
        hparams['custom_counts'] = None
    

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset in vars(datasets):
        train_dataset = vars(datasets)[args.dataset](args.data_dir, 'train', hparams)
        val_dataset = vars(datasets)[args.dataset](args.data_dir, 'val', hparams)
        test_dataset = vars(datasets)[args.dataset](args.data_dir, 'test', hparams)

        # ─────────────────────────────────────────────────────────────────────────
        # [NEW] --custom_counts 로 지정된 개수만큼 *train* split 다운샘플링
        # (도메인 수 × 클래스 수 행렬)
        # ─────────────────────────────────────────────────────────────────────────
        if hparams.get('custom_counts'):
            c_counts = hparams['custom_counts']
            assert len(c_counts) == len(train_dataset), \
                f"custom_counts: 도메인 수가 {len(train_dataset)}인데 {len(c_counts)}개가 전달됨"

            for env_i in range(len(train_dataset)):
                env_ds = train_dataset[env_i]          # 각 도메인의 Dataset
                keep_per_cls = c_counts[env_i]
                num_classes = len(keep_per_cls)

                # 라벨 벡터 획득
                if hasattr(env_ds, "targets"):
                    labels_np = np.asarray(env_ds.targets).astype(int)
                elif hasattr(env_ds, "tensors"):
                    labels_np = env_ds.tensors[1].numpy().astype(int)
                else:
                    raise AttributeError("Dataset에 targets 배열이 없습니다.")

                assert num_classes == labels_np.max() + 1, \
                    "custom_counts 내부 리스트 길이(클래스 수)가 실제 클래스 수와 다릅니다."

                sel_idx = []
                for cls, n_keep in enumerate(keep_per_cls):
                    cls_idx = np.where(labels_np == cls)[0]
                    assert n_keep <= len(cls_idx), \
                        f"env{env_i}-class{cls}: 요청 {n_keep} > 보유 {len(cls_idx)}"
                    np.random.shuffle(cls_idx)
                    sel_idx.extend(cls_idx[:n_keep])

                # 다운샘플 적용
                train_dataset.datasets[env_i] = SubsetWithTargets(env_ds, sel_idx)

    else:
        raise NotImplementedError

    # --- [ TALLY 통합 코드 추가 시작 ] ---
    if args.algorithm == 'TALLYAlgorithm':
        # BoDA 데이터셋을 TallyDatasetWrapper로 감싸기
        train_dataset_wrapped = TallyDatasetWrapper(train_dataset, hparams)
    # --- [ TALLY 통합 코드 추가 끝 ] ---   

    num_workers = train_dataset.N_WORKERS
    input_shape = train_dataset.input_shape
    num_classes = train_dataset.num_classes
    n_steps = args.steps or train_dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or train_dataset.CHECKPOINT_FREQ
    many_shot_thr = train_dataset.MANY_SHOT_THRES
    few_shot_thr = train_dataset.FEW_SHOT_THRES

    if args.selected_envs is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, args.selected_envs)
        val_dataset = torch.utils.data.Subset(val_dataset, args.selected_envs)
        test_dataset = torch.utils.data.Subset(test_dataset, args.selected_envs)
    env_ids = args.selected_envs if args.selected_envs is not None else np.arange(len(train_dataset))

    print("Dataset:")

    '''
    # --------------------------------------------------------------------- #
    # [새] ① 도메인별 train class 분포 / ② pairwise KL / ③ train-vs-test KL #
    # --------------------------------------------------------------------- #
    from mdlt.utils.misc import kl_divergence
    train_cls_cnt, test_cls_cnt = {}, {}

    header = ['env'] + [f'c{c}' for c in range(num_classes)] + ['total']
    misc.print_row(header, colwidth=8)

    for i, (tr, _, te) in enumerate(zip(train_dataset, val_dataset, test_dataset)):
        t_tr = tr.targets if 'Imbalance' not in args.dataset else tr.tensors[1].numpy()
        t_te = te.targets if 'Imbalance' not in args.dataset else te.tensors[1].numpy()
        cnt_tr = np.bincount(t_tr, minlength=num_classes)
        cnt_te = np.bincount(t_te, minlength=num_classes)
        train_cls_cnt[f'env{env_ids[i]}'] = cnt_tr
        test_cls_cnt[f'env{env_ids[i]}']  = cnt_te

        misc.print_row([f'env{env_ids[i]}'] + cnt_tr.tolist() + [cnt_tr.sum()], colwidth=8)

    # ――― ① domain 간 KL(train) ―――
    print("\n[KL(train env_i ‖ train env_j)]")
    for i in range(len(env_ids)):
        for j in range(i + 1, len(env_ids)):
            ei, ej = f'env{env_ids[i]}', f'env{env_ids[j]}'
            kl = kl_divergence(train_cls_cnt[ei], train_cls_cnt[ej])
            print(f'{ei} ‖ {ej}: {kl:.4f}')

    # ――― ② train vs test KL per env ―――
    print("\n[KL(train ‖ test) per env]")
    for i in range(len(env_ids)):
        env = f'env{env_ids[i]}'
        kl = kl_divergence(train_cls_cnt[env], test_cls_cnt[env])
        print(f'{env}: {kl:.4f}')


    for i, (tr, va, te) in enumerate(zip(train_dataset, val_dataset, test_dataset)):
        print(f"\tenv{env_ids[i]}:\t{len(tr)}\t|\t{len(va)}\t|\t{len(te)}")
    
    '''

    # Split each env into train, val, test
    train_splits, val_splits, test_splits = [], [], []
    train_labels = dict()
    for i, env in enumerate(zip(train_dataset, val_dataset, test_dataset)):
        env_train, env_val, env_test = env
        
        if hparams['class_balanced']:
            train_weights = misc.make_balanced_weights_per_sample(
                env_train.targets if 'Imbalance' not in args.dataset else env_train.tensors[1].numpy())
            val_weights = misc.make_balanced_weights_per_sample(
                env_val.targets if 'Imbalance' not in args.dataset else env_val.tensors[1].numpy())
            test_weights = misc.make_balanced_weights_per_sample(
                env_test.targets if 'Imbalance' not in args.dataset else env_test.tensors[1].numpy())
        else:
            train_weights, val_weights, test_weights = None, None, None
        train_splits.append((env_train, train_weights))
        val_splits.append((env_val, val_weights))
        test_splits.append((env_test, test_weights))
        train_labels[f"env{env_ids[i]}"] = env_train.targets if 'Imbalance' not in args.dataset else env_train.tensors[1].numpy()

    # --- [ TALLY 통합 코드 수정 시작 ] ---
    if args.algorithm == 'TALLYAlgorithm':
        # TALLY는 warmup과 train 단계에서 다른 로더/이터레이터를 사용합니다.
        train_loader_warmup = DataLoader(
            dataset=train_dataset_wrapped,
            batch_size=hparams['batch_size'],
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True
        )
        
        # iter_num은 한 epoch당 step 수와 유사한 개념입니다.
        steps_per_epoch = len(train_dataset_wrapped) // hparams['batch_size']
        tally_sampler = TallySampler(
            train_dataset_wrapped, 
            batch_size=hparams['batch_size'] // 2, 
            iter_num=steps_per_epoch
        )
        train_loader_tally = DataLoader(
            dataset=train_dataset_wrapped,
            batch_size=hparams['batch_size'], # sampler가 2 * (batch_size/2) 만큼 인덱스를 반환
            num_workers=num_workers,
            sampler=tally_sampler,
            pin_memory=True
        )
        
        # 학습 루프에서 사용할 수 있도록 이터레이터로 변환
        warmup_iterator = iter(train_loader_warmup)
        tally_iterator = iter(train_loader_tally)

    else: # 다른 모든 알고리즘은 기존 방식 사용
        train_loaders = [InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=hparams['batch_size'],
            num_workers=num_workers)
            for env, env_weights in train_splits
        ]
        train_minibatches_iterator = zip(*train_loaders)
    # --- [ TALLY 통합 코드 수정 끝 ] ---

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=num_workers)
        for env, _ in (val_splits + test_splits)
    ]
    # loader for online training feature updates
    train_feat_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=num_workers)
        for env, _ in train_splits
    ] if 'BoDA' in args.algorithm else None
    eval_weights = [None for _, weights in (val_splits + test_splits)]
    eval_loader_names = [f'env{env_ids[i]}_val' for i in range(len(val_splits))]
    eval_loader_names += [f'env{env_ids[i]}_test' for i in range(len(test_splits))]
    feat_loader_names = [f'env{env_ids[i]}' for i in range(len(train_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)



    # --- [ 수정 시작 ] ---
    if args.algorithm == 'TALLYAlgorithm':
        # TALLYDataset은 여러 도메인을 포함하는 단일 객체일 수 있음
        # train_dataset.datasets는 각 도메인별 Subset 리스트
        # TALLY의 update_statistics는 통합된 데이터셋 객체에서 호출되어야 함
        # 프로젝트의 dataset 구성에 따라 train_dataset 또는 train_dataset.raw_dataset 등을 넘겨야 할 수 있음
        algorithm = algorithm_class(
            input_shape, num_classes, len(train_dataset), hparams, train_dataset=train_dataset_wrapped
        )
    elif args.algorithm == 'MLIR':
        algorithm = algorithm_class(train_dataset.input_shape, train_dataset.num_classes, len(train_dataset.ENVIRONMENTS), hparams)   
     
    # --- [ 수정 끝 ] ---

    # load stage1 model if using 2-stage algorithm
    if 'CRT' in args.algorithm:
        args.pretrained = os.path.join(
            args.output_dir.replace(args.output_folder_name, args.stage1_folder), hparams['stage1_model']
        ).replace(args.algorithm, args.stage1_algo)
        print(args.pretrained)
        
        args.pretrained = args.pretrained.replace(
            f"seed{args.pretrained[args.pretrained.find('seed') + len('seed')]}", 'seed0')
        print(args.pretrained)
        
        assert os.path.isfile(args.pretrained)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_dict'].items():
            if 'classifier' not in k and 'network.1.' not in k:
                new_state_dict[k] = v
        algorithm.load_state_dict(new_state_dict, strict=False)
        print(f"===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]")
        print(f"===> Pre-trained model loaded: '{args.pretrained}'")

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_step = checkpoint['start_step']
            args.best_val_acc = checkpoint['best_val_acc']
            algorithm.load_state_dict(checkpoint['model_dict'])
            print(f"===> Loaded checkpoint '{args.resume}' (step [{start_step}])")
        else:
            print(f"===> No checkpoint found at '{args.resume}'")

    algorithm.to(device)

    # train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env, _ in train_splits])

    def save_checkpoint(best=False, filename='model.pkl', curr_step=0):
        if args.skip_model_save:
            return
        filename = os.path.join(args.output_dir, filename)
        save_dict = {
            "args": vars(args),
            "best_val_acc": args.best_val_acc,
            "start_step": curr_step + 1,
            "num_classes": num_classes,
            "num_domains": len(train_dataset),
            "model_input_shape": input_shape,
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, filename)
        if best:
            shutil.copyfile(filename, filename.replace('pkl', 'best.pkl'))

    last_results_keys = None
    for step in tqdm(range(start_step, n_steps), total=n_steps):
        step_start_time = time.time()

        # --- [ TALLY 통합 수정 ] ---
        # 알고리즘에 따라 배치를 가져오는 방식을 다르게 합니다.
        if args.algorithm == 'TALLYAlgorithm':
            warmup_steps = hparams.get('warmup_steps', 2000)
            
            try:
                if step < warmup_steps:
                    minibatch_tuple = next(warmup_iterator)
                else:
                    minibatch_tuple = next(tally_iterator)
            except StopIteration:
                # 이터레이터가 끝나면 새로 만듭니다.
                if step < warmup_steps:
                    warmup_iterator = iter(train_loader_warmup)
                    minibatch_tuple = next(warmup_iterator)
                else:
                    tally_iterator = iter(train_loader_tally)
                    minibatch_tuple = next(tally_iterator)
            
            # TALLY는 7개 항목을 포함하는 단일 튜플 배치를 처리합니다.
            minibatches_device = tuple(tensor.to(device) for tensor in minibatch_tuple)
        else:
            # 다른 알고리즘들은 각 도메인에서 온 (x, y) 튜플의 리스트를 처리합니다.
            minibatches_device = [(x.to(device), y.to(device))
                                  for x, y in next(train_minibatches_iterator)]
        # --- [ TALLY 통합 수정 끝 ] ---

        # update features before training step
        train_features = {}
        if 'BoDA' in args.algorithm and (step > 0 and step % hparams["feat_update_freq"] == 0):
            curr_tr_feats, curr_tr_labels = collections.defaultdict(list), collections.defaultdict(list)
            for name, loader in sorted(zip(feat_loader_names, train_feat_loaders), key=lambda x: x[0]):
                algorithm.eval()
                with torch.no_grad():
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        feats = algorithm.return_feats(x)
                        curr_tr_feats[name].extend(feats.data)
                        curr_tr_labels[name].extend(y.data)
            train_features = {'feats': curr_tr_feats, 'labels': curr_tr_labels}

        algorithm.train()

        step_vals = algorithm.update(minibatches_device, train_features)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            class_acc_output = collections.defaultdict(list)
            shot_acc_output = collections.defaultdict(list)
            env_acc_output = {}
            for name, loader, weights in sorted(evals, key=lambda x: x[0]):
                if 'test' in name:
                    acc, shot_acc, class_acc = misc.accuracy(
                        algorithm, loader, weights, train_labels[name.split('_')[0]],
                        many_shot_thr, few_shot_thr, device, class_shot_acc=True)
                    class_acc_output[name.split('_')[0]] = list(class_acc)
                    env_acc_output[name.split('_')[0]] = acc
                    shot_acc_output['many'].extend(shot_acc[0])
                    shot_acc_output['median'].extend(shot_acc[1])
                    shot_acc_output['few'].extend(shot_acc[2])
                    shot_acc_output['zero'].extend(shot_acc[3])
                else:
                    acc = misc.accuracy(algorithm, loader, weights, train_labels[name.split('_')[0]],
                                        many_shot_thr, few_shot_thr, device, class_shot_acc=False)
                results[name] = acc

            # shot-wise results
            for shot in ['many', 'median', 'few', 'zero']:
                if len(shot_acc_output[shot]) == 0:
                    shot_acc_output[shot].append(-1)
                results[f"sht_{shot}"] = np.mean(shot_acc_output[shot])

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = list(results.keys())
            if results_keys != last_results_keys:
                print("\n")
                misc.print_row([key for key in results_keys if key not in {'mem_gb', 'step_time'}], colwidth=8)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys if key not in {'mem_gb', 'step_time'}], colwidth=8)

            results.update({
                'hparams': hparams,
                'args': vars(args),
                'class_acc': class_acc_output
            })

            epochs_path = os.path.join(args.output_dir, 'results.json')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            # record best validation accuracy (mean over all envs)
            val_env_keys = [f'env{i}_val' for i in env_ids if f'env{i}_val' in results.keys()]
            val_acc_mean = np.mean([results[key] for key in val_env_keys])
            is_best = val_acc_mean > args.best_val_acc
            args.best_val_acc = max(val_acc_mean, args.best_val_acc)
            if is_best:
                best_class_acc = class_acc_output
                best_env_acc = env_acc_output
                best_shot_acc = {s: np.mean(shot_acc_output[s]) for s in ['many', 'median', 'few', 'zero']}

            save_checkpoint(best=is_best, curr_step=step)

            # tensorboard logger
            for key in checkpoint_vals.keys() - {'step_time'}:
                tb_logger.log_value(key, results[key], step)
            tb_logger.log_value('val_acc', val_acc_mean, step)
            tb_logger.log_value('test_acc_mean', np.mean(list(env_acc_output.values())), step)
            tb_logger.log_value('test_acc_worst', min(env_acc_output.values()), step)
            for i in env_ids:
                tb_logger.log_value(f'test_env{i}_acc', results[f"env{i}_test"], step)
            for s in ['many', 'median', 'few', 'zero']:
                tb_logger.log_value(f'shot_{s}', results[f"sht_{s}"], step)
            if hasattr(algorithm, 'optimizer'):
                tb_logger.log_value('learning_rate', algorithm.optimizer.param_groups[0]['lr'], step)

            checkpoint_vals = collections.defaultdict(lambda: [])

    print("\nTest accuracy (best validation checkpoint):")
    print(f"\tmean:\t[{np.mean(list(best_env_acc.values())):.3f}]\n\tworst:\t[{min(best_env_acc.values()):.3f}]")
    print("Shot-wise accuracy:")
    for s in ['many', 'median', 'few', 'zero']:
        print(f"\t[{s[:4]}]:\t[{best_shot_acc[s]:.3f}]")
    print("Class-wise accuracy:")
    for env in sorted(best_class_acc):
        print('\t[{}] overall {:.3f}, class-wise {}'.format(
            env, best_env_acc[env], (np.array2string(
                np.array(best_class_acc[env]), separator=', ', formatter={'float_kind': lambda x: "%.3f" % x}))))

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
        
        
    # hyunggyu - for time & gpu_usage logging
    # 경로 설정 (예: ./output/Algo_Dataset_hparamsX_seedY/gpu_summary.txt)
    # resource_log_path = os.path.join(args.output_dir, "resource_summary.txt")

    # # 추적 종료 및 결과 저장
    # tracker.stop()
    # tracker.save(resource_log_path)
