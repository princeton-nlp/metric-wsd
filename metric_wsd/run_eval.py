import os
import sys
import argparse
import yaml

from tqdm import tqdm

import torch

from metric_wsd.utils.data_loader import WSDDataLoader
from metric_wsd.run import MetricWSD
from metric_wsd.utils.utils import (
    args_factory,
    load_args,
    save_pred_file,
    evaluate_from_pred_file,
    load_ckpt
)
from metric_wsd.config import Config

config = Config()


def model_inference(args):
    model, ckpt_args, _ = load_ckpt(load_path=os.path.join('experiments', args.dir, args.name))
    model.cuda()
    ckpt_args.batch_size = 1
    ckpt_args.debug = False
    if args.max_inference_supports is not None:
        ckpt_args.max_inference_supports = args.max_inference_supports

    dm = WSDDataLoader(ckpt_args)

    if args.evalsplit == 'SE07':
        data = dm.dev
        gold_key_path = os.path.join(config.SE07[0], f'{config.SE07[1]}.gold.key.txt')

        predictions = []
        for i, batch in enumerate(data):
            examplekey = batch['examplekey']
            _, preds = model.forward_eval(batch, device=torch.device('cuda'))
            predictions += preds
        save_pred_file(predictions)
        f1 = evaluate_from_pred_file(gold_key_path)
        print(f1)

    elif args.evalsplit == 'ALL':
        data = dm.test
        predictions = []
        gold_key_path = os.path.join(config.ALL[0], f'{config.ALL[1]}.gold.key.txt')

        for i, batch in enumerate(data):
            if i % 500 == 0:
                print(f'{i}th test example processed.')
            targetword = batch['targetword']
            examplekey = batch['examplekey']
            evalset_name = examplekey.split('.')[0]
            pos = batch['targetword'].split('+')[1]

            _, preds = model.forward_eval(batch, device=torch.device('cuda'))
            predictions += preds
            
        save_pred_file(predictions)
        f1 = evaluate_from_pred_file(gold_key_path)
        print(f'Evaluating {gold_key_path.split("/")[-1]}')
        print(f1)

    else:
        raise ValueError('Invalid eval set.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler="resolve")
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--evalsplit", type=str, default='SE07', help='[SE07 | ALL]')
    parser.add_argument("--max_inference_supports", type=int, default=None)
    args = parser.parse_args()

    model_inference(args)