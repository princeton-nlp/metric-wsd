import os
import csv
import random
import subprocess
from collections import Counter
from argparse import Namespace
import glob
import yaml

import torch
from torch.optim import AdamW

from metric_wsd.models.models import CBERTLinear, CBERTProto, PairCBERTProto, CBERTProtoConcat
from metric_wsd.config import Config

config = Config()


def update_args(args):
    """Make args compatible with different models and runs."""
    if not hasattr(args, 'dist'):
        setattr(args, 'dist', 'dot')
    if not hasattr(args, 'ks_support_kq_query'):
        setattr(args, 'ks_support_kq_query', None)
    if not hasattr(args, 'global_step'):
        setattr(args, 'global_step', 0)
    if not hasattr(args, 'sample_strategy_threshold'):
        setattr(args, 'sample_strategy_threshold', 0)
    return args


def get_model_class(args):
    if args.model_type == 'cbert-linear':
        model_class = CBERTLinear
    elif args.model_type == 'cbert-proto':
        if args.dist in ('dot', 'l2'):
            model_class = CBERTProto
        elif args.dist == 'concat':
            model_class = CBERTProtoConcat
        elif args.dist.startswith('text'):
            model_class = PairCBERTProto
    else:
        raise ValueError('Model type not implemented.')
    return model_class


def save_ckpt(model, args, f1, epoch, global_step, optimizer, latest=False):
    if not latest:
        args.best_ckpt_path = args.filepath.format(epoch=epoch, f1=f1)
        args.best_model_score = f1
        checkpoint = {'ckpt_path': args.best_ckpt_path}
    else:
        checkpoint = {'ckpt_path': os.path.join(args.ckpt_dir, 'latest.ckpt')}

    args.current_epoch = epoch
    args.global_step = global_step
    checkpoint['args'] = vars(args)
    checkpoint['states'] = model.state_dict()
    checkpoint['optimizer_states'] = optimizer.state_dict()

    if not latest:
        for rm_path in glob.glob(os.path.join(args.ckpt_dir, '*.pt')):
            os.remove(rm_path)

    torch.save(checkpoint, checkpoint['ckpt_path'])
    print(f"Model saved at: {checkpoint['ckpt_path']}")


def load_ckpt(load_path):
    checkpoint = torch.load(load_path, map_location='cpu')
    args = Namespace(**checkpoint['args'])
    args = update_args(args)
    states = checkpoint['states']
    model_class = get_model_class(args)
    model = model_class(args)
    model.load_state_dict(states)
    model.eval()
    print('Model loaded from:', load_path)

    if 'optimizer_states' in checkpoint:
        optimizer = AdamW(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint['optimizer_states'])
        return model, args, optimizer
    else:
        return model, args, None


def save_args(args):
    with open(args.args_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f'Arg file saved at: {args.args_path}')


def load_args(ckpt_dir, as_namespace):
    path = os.path.join(ckpt_dir, 'args.yaml')
    with open(path) as f:
        args = yaml.safe_load(f)
    if as_namespace:
        args = Namespace(**args)
    return args


def args_factory(args):
    args.filename = 'model-epoch={epoch:03d}-f1={f1:.1f}.pt'
    args.ckpt_dir = os.path.join(config.EXP_DIR, args.run_name)
    args.filepath = os.path.join(args.ckpt_dir, args.filename)
    args.args_path = os.path.join(args.ckpt_dir, 'args.yaml')

    if args.model_type == 'cbert-proto':
        assert args.episodic

    if args.episodic:
        args.accumulate_grad_batches, args.batch_size = args.batch_size, 1
    else:
        args.accumulate_grad_batches = 1
    
    args.num_sanity_val_steps = 0
    args.progress_bar_refresh_rate = 0
    return args


def sample_examples(keys, weights, k, seed):
    if seed is not None:
        random.seed(seed)
    sampled_keys = random.choices(keys, weights=weights, k=k)  # sample with replacement
    return dict(Counter(sampled_keys))


def save_pred_file(predictions, args=None, filename=None):
    filename = 'tmp.key.txt' if filename is None else filename
    if args is None:
        pred_key_path = os.path.join(config.TMP_DIR, 'tmp.key.txt')
    else:
        pred_key_path = os.path.join(args.ckpt_dir, 'tmp.key.txt')
    with open(pred_key_path, 'w') as f:
        for pred in predictions:
            f.write(f'{pred}\n')


def evaluate_from_pred_file(gold_key_path, args=None, filename=None):
    filename = 'tmp.key.txt' if filename is None else filename
    if args is None:
        pred_key_path = os.path.join(config.TMP_DIR, filename)
    else:
        pred_key_path = os.path.join(args.ckpt_dir, filename)
    eval_cmd = ['java', 'Scorer', gold_key_path, pred_key_path]
    os.chdir(config.SCORER_DIR)
    output = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE).communicate()[0]
    output = str(output, 'utf-8')
    output = output.splitlines()
    _, _, f1 =  [float(output[i].split('=')[-1].strip()[:-1]) for i in range(3)]
    return f1


def load_glosses(path):
    sensekey_to_gloss = {}
    with open(path) as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for line in csv_reader:
            sensekey_to_gloss[line[0]] = line[4]
    return sensekey_to_gloss