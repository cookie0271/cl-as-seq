import os
import os.path as path
import shutil
import socket
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from modulefinder import ModuleFinder

import math
import random
import torch
import yaml
from einops import rearrange, pack
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import DATASET
from models import MODEL
from train import forward
from utils import Timer

parser = ArgumentParser()
parser.add_argument('--model-config', '-mc', required=True)
parser.add_argument('--data-config', '-dc', required=True)
parser.add_argument('--log-dir', '-l')
parser.add_argument('--override', '-o', default='')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_config(config_path):
    with open(config_path, 'r') as f:
        new_config = yaml.full_load(f)
    config = {}
    if 'include' in new_config:
        include_config = get_config(new_config['include'])
        config.update(include_config)
        del new_config['include']
    config.update(new_config)
    return config

def load_state_dict_flexible(model, state_dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    if any(k.startswith('module.') for k in state_dict):
        stripped = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(stripped)
    else:
        prefixed = {f'module.{k}': v for k, v in state_dict.items()}
        model.load_state_dict(prefixed)



def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    if torch.cuda.is_available():
        print(f'Running on {socket.gethostname()} | {torch.cuda.device_count()}x {torch.cuda.get_device_name()}')
    args = parser.parse_args()

    # Load config
    config = get_config(args.model_config)
    data_config = get_config(args.data_config)
    config.update(data_config)

    # Override options
    for option in args.override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                here[key] = {}
            here = here[key]
        if keys[-1] not in here:
            print(f'Warning: {address} is not defined in config file.')
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)

    if 'y_vocab' in config and config['y_vocab'] is None:
        config['y_vocab'] = config['tasks']

    set_seed(config.get('seed', 0))

    # Prevent overwriting
    config['log_dir'] = args.log_dir

    # Get a free port for DDP
    sock = socket.socket()
    sock.bind(('', 0))
    ddp_port = sock.getsockname()[1]
    sock.close()

    # Start distributed evaluation only when multiple GPUs are available
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError('CUDA is required for meta_train_score.py. Found 0 CUDA devices.')

    distributed = world_size > 1
    if distributed:
        assert config['batch_size'] % world_size == 0, 'Batch size must be divisible by the number of GPUs.'
        config['batch_size'] //= world_size
        assert config['eval_batch_size'] % world_size == 0, 'Eval batch size must be divisible by the number of GPUs.'
        config['eval_batch_size'] //= world_size
        mp.spawn(train, args=(world_size, ddp_port, args, config, distributed), nprocs=world_size)
    else:
        train(0, 1, ddp_port, args, config, distributed)


def train(rank, world_size, port, args, config, distributed=True):
    if distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

        # Initialize process group
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Build model
    device = torch.device('cuda', rank)
    model = MODEL[config['model']](config).to(device)
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[rank],
            find_unused_parameters=config.get('ddp_find_unused_parameters', True),
        )
    optim = getattr(torch.optim, config['optim'])(model.parameters(), **config['optim_args'])
    lr_sched = getattr(lr_scheduler, config['lr_sched'])(optim, **config['lr_sched_args'])

    # Resume checkpoint
    if distributed:
        ckpt_path_holder = [None]
        if rank == 0:
            ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
            ckpt_path_holder[0] = ckpt_paths[-1] if len(ckpt_paths) > 0 else None
        dist.broadcast_object_list(ckpt_path_holder, src=0)
        ckpt_path = ckpt_path_holder[0]
    else:
        ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
        ckpt_path = ckpt_paths[-1] if len(ckpt_paths) > 0 else None

    if ckpt_path is None:
        raise RuntimeError(f'No checkpoint found in {config["log_dir"]}')

    # Get step number from checkpoint name
    start_step = int(path.basename(ckpt_path).split('-')[1].split('.')[0])
    if start_step != config['max_train_steps']:
        raise RuntimeError(f'Latest checkpoint {ckpt_path} does not match max_train_steps {config["max_train_steps"]}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    load_state_dict_flexible(model, ckpt['model'])
    optim.load_state_dict(ckpt['optim'])
    lr_sched.load_state_dict(ckpt['lr_sched'])
    print(f'Checkpoint loaded from {ckpt_path}')
    optim.zero_grad()

    # Data
    Dataset = DATASET[config['dataset']]
    train_set = Dataset(config, root='./data', meta_split='train')
    train_loader = DataLoader(
        train_set,
        batch_size=config['eval_batch_size'],
        num_workers=config['num_workers'])
    train_loader_iter = iter(train_loader)

    start_time = datetime.now()
    print(f'Computation started at {start_time}')

    # Meta-test
    model.eval()
    meta_train_scores = {}
    with torch.no_grad(), Timer('Evaluation time: {:.3f}s'):
        loss_mean = 0
        correct, total = 0, 0
        eval_size = config['eval_iters'] * config['eval_batch_size']
        for _ in range(config['eval_iters']):
            train_x, train_y, test_x, test_y = next(train_loader_iter)
            train_x, train_y = train_x.to(device), train_y.to(device)
            test_x, test_y = test_x.to(device), test_y.to(device)

            batch_size = train_x.shape[0]
            digested = 0
            while batch_size - digested > 0:
                bite = min(batch_size - digested, math.ceil(config['eval_batch_size'] / config['num_bites']))
                train_x_bite = train_x[digested:digested + bite].to(rank)
                train_y_bite = train_y[digested:digested + bite].to(rank)
                test_x_bite = test_x[digested:digested + bite].to(rank)
                test_y_bite = test_y[digested:digested + bite].to(rank)

                output = forward(
                    model, train_x_bite, train_y_bite, test_x_bite, test_y_bite,
                    eval_size=eval_size)

                digested += bite

                loss_mean += output['loss_mean'] * output['proportion']
                if 'correct' in output:
                    correct += output['correct']
                    total += output['total']

        if distributed:
            gathered_loss_mean = torch.zeros(world_size, dtype=loss_mean.dtype, device=loss_mean.device)
            dist.all_gather_into_tensor(gathered_loss_mean, loss_mean)
            loss_mean = gathered_loss_mean.mean().item()
        else:
            loss_mean = loss_mean.item()
        if rank == 0:
            meta_train_scores['loss/train'] = loss_mean

        if total > 0:
            if distributed:
                gathered_correct = torch.zeros(world_size, dtype=correct.dtype, device=correct.device)
                gathered_total = torch.zeros(world_size, dtype=total.dtype, device=total.device)
                dist.all_gather_into_tensor(gathered_correct, correct)
                dist.all_gather_into_tensor(gathered_total, total)
                acc_train = (gathered_correct.sum() / gathered_total.sum()).item()
            else:
                acc_train = (correct / total).item()
            if rank == 0:
                meta_train_scores['acc/train'] = acc_train

    model.train()

    if rank == 0:
        end_time = datetime.now()
        print()
        print(f'Evaluation ended at {end_time}')
        print(f'Elapsed time: {end_time - start_time}')
        print(f'loss/train: {meta_train_scores["loss/train"]:.4f}')
        if 'acc/train' in meta_train_scores:
            print(f'acc/train: {meta_train_scores["acc/train"]:.4f}')
        torch.save(meta_train_scores, path.join(config['log_dir'], 'meta_train_scores.pt'))

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
