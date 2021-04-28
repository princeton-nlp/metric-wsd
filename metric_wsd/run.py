import os
import sys
from pathlib import Path
import argparse
from argparse import Namespace
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from transformers import get_linear_schedule_with_warmup

from metric_wsd.utils.data_loader import WSDDataLoader
from metric_wsd.utils.utils import (
    args_factory,
    save_args,
    save_pred_file,
    evaluate_from_pred_file,
    save_ckpt,
    load_ckpt,
    get_model_class
)
from metric_wsd.config import Config

config = Config()


class LoadCheckpointCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        pl_module.current_epoch = pl_module.args.current_epoch
        trainer.current_epoch = pl_module.args.current_epoch
        trainer.global_step = pl_module.args.global_step


class CheckpointCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        f1 = trainer.callback_metrics.get('f1')
        if f1 > pl_module.best_model_score:
            save_ckpt(
                model=pl_module.model,
                args=pl_module.args,
                f1=f1,
                epoch=pl_module.current_epoch,
                global_step=pl_module.global_step,
                optimizer=trainer.optimizers[0]
            )
            pl_module.best_model_score = f1

        save_ckpt(
            model=pl_module.model,
            args=pl_module.args,
            f1=f1,
            epoch=pl_module.current_epoch,
            global_step=pl_module.global_step,
            optimizer=trainer.optimizers[0],
            latest=True
        )


class PrintingCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if batch_idx % pl_module.args.print_every == 0 and batch_idx != 0:
            train_acc = trainer.callback_metrics['train_acc']
            train_loss = trainer.callback_metrics['train_loss']
            print(f'[train] Epoch: {pl_module.current_epoch} | batch: {batch_idx} | train acc: {train_acc:.2f} | train_loss: {train_loss:.4f}')

    def on_validation_end(self, trainer, pl_module):
        f1 = trainer.callback_metrics.get('f1')
        print(f'[dev] Epoch: {pl_module.current_epoch} | f1: {f1}')
        print('==========')


class MetricWSD(pl.LightningModule):
    def __init__(self, args, resume_model=None, resume_optimizer=None):
        super().__init__()
        if isinstance(args, dict):
            # handle ckpt loading
            args = Namespace(**args)
        self.args = args

        if resume_model is None:
            model_class = get_model_class(args)
            self.model = model_class(args=args)
            self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr)
            self.best_model_score = float('-inf')
            self.total_seen = 0
        else:
            self.model = resume_model
            self.optimizer = resume_optimizer
            self.total_seen = args.total_seen
            self.best_model_score = args.best_model_score
            self.global_step = args.global_step
    
    def forward(self, batch, mode):
        if mode == 'train':
            return self.model(batch, self.device)
        elif mode == 'dev':
            return self.model.forward_eval(batch, self.device)

    def training_step(self, batch, batch_idx):
        if self.args.gpu_stats:
            os.system('nvidia-smi')
        loss, correct = self(batch, mode='train')
        acc = correct.sum().float() / correct.size(0)
        
        self.total_seen += correct.size(0)
        self.args.total_seen = self.total_seen

        tqdm_dict = {'train_loss': loss.item(), 'train_acc': acc.item(), 'total_seen': self.total_seen, 'num_examples': correct.size(0)}
        return OrderedDict({
            'train_acc': acc,
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'total_seen': self.total_seen,
            'num_examples': correct.size(0)
        })

    def validation_step(self, batch, batch_idx):
        correct, preds = self(batch, mode='dev')
        return OrderedDict({'correct': correct, 'preds': preds})

    def validation_end(self, outputs):
        correct = []
        preds = []
        for output in outputs:
            if self.args.gpus <= 1:
                correct += output['correct']
                preds += output['preds']
            else:
                # covert the returns by distributed training from [(True,)] to [True]
                correct += [output['correct'][0][0]]
                preds += [output['preds'][0][0]]
        dev_acc = sum(correct) / len(correct)

        save_pred_file(preds, self.args)
        f1 = evaluate_from_pred_file(config.SE07_GOLD_KEY_PATH, self.args)

        tqdm_dict = {'dev_acc': dev_acc, 'f1': f1, 'total_seen': self.total_seen}
        return {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'f1': f1, 'total_seen': self.total_seen}

    def configure_optimizers(self):
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        return self.optimizer


def main(args):
    args = args_factory(args)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    save_args(args)

    dm = WSDDataLoader(args)

    args.wn_senses = dm.wn_senses
    args.training_words = dm.training_words

    logger = WandbLogger(name=args.run_name, project=config.PROJECT_NAME) if args.wandb else None

    metric_wsd_pl_module = MetricWSD(args)
    print(f'Trainable params: {sum(p.numel() for p in metric_wsd_pl_module.parameters() if p.requires_grad)}')
    print(f'All params      : {sum(p.numel() for p in metric_wsd_pl_module.parameters())}')

    trainer = pl.Trainer(gpus=args.gpus,
                         num_nodes=args.num_nodes,
                         distributed_backend='dp',
                         num_sanity_val_steps=0,
                         checkpoint_callback=False,
                         max_epochs=args.max_epochs,
                         logger=logger,
                         profiler=args.profiler,
                         progress_bar_refresh_rate=args.progress_bar_refresh_rate,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         callbacks=[PrintingCallback(), CheckpointCallback()])
    trainer.fit(metric_wsd_pl_module, dm.train, dm.dev)

    model, args, optimizer = load_ckpt(metric_wsd_pl_module.args.best_ckpt_path)


def resume(args):
    path = os.path.join('experiments', args.run_name, args.resume_from_ckpt)
    resume_model, ckpt_args, resume_optimizer = load_ckpt(path)
    resume_model.train()
    metric_wsd_pl_module = MetricWSD(ckpt_args,
                                     resume_model=resume_model,
                                     resume_optimizer=resume_optimizer)

    dm = WSDDataLoader(ckpt_args)
    logger = WandbLogger(name=args.run_name, project=config.PROJECT_NAME) if args.wandb else None

    print(f'Trainable params: {sum(p.numel() for p in metric_wsd_pl_module.parameters() if p.requires_grad)}')
    print(f'All params      : {sum(p.numel() for p in metric_wsd_pl_module.parameters())}')

    trainer = pl.Trainer(gpus=args.gpus,
                         num_nodes=args.num_nodes,
                         distributed_backend='dp',
                         num_sanity_val_steps=0,
                         checkpoint_callback=False,
                         max_epochs=ckpt_args.max_epochs,
                         logger=logger,
                         profiler=args.profiler,
                         progress_bar_refresh_rate=ckpt_args.progress_bar_refresh_rate,
                         accumulate_grad_batches=ckpt_args.accumulate_grad_batches,
                         callbacks=[PrintingCallback(), CheckpointCallback(), LoadCheckpointCallback()])
    trainer.fit(metric_wsd_pl_module, dm.train, dm.dev)

    model, args, optimizer = load_ckpt(metric_wsd_pl_module.args.best_ckpt_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0], conflict_handler="resolve")
    parser.add_argument("--model-type", type=str, required=True, help="[cbert-linear | cbert-proto]")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--freeze-context-enc", action="store_true")

    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--gpu-stats", action="store_true")
    parser.add_argument("--resume-from-ckpt", type=str, default=None, help="Name of the model checkpoint.")
    parser.add_argument("--print-every", type=int, default=500)

    parser.add_argument("--episodic", action="store_true")
    parser.add_argument("--episodic_k", type=int, default=40)
    parser.add_argument("--support_query_ratio", type=float, default=0.4)
    parser.add_argument("--sampling", type=str, default="balanced", help="[balanced (Pb) | uniform (Pu)]")
    parser.add_argument("--ks_support_kq_query", nargs='+', type=int, default=[3, 50, 5],
                        help="Pos 1: Number of supports to select per sense. "
                             "Pos 2: Max number of queries for each sense. "
                             "Pos 3: Max number of senses per word.")
    parser.add_argument("--sample-strategy-threshold", type=int, default=0,
                        help="Threshold on word frequency. Lower than threshold uses Ratio Split. Higher than threshold uses Max Query."
                             "Min: 0. Max: 16000. Set to 0 means always use Max Query. 16000 always use Ratio Split.")
    parser.add_argument("--max_inference_supports", type=int, default=-1)
    parser.add_argument("--mix-strategy", action="store_true")
    parser.add_argument("--max_senses", type=int, default=5)
    parser.add_argument("--max_queries", type=int, default=40)
    parser.add_argument("--avg_num_supports", type=int, default=3)
                    
    parser.add_argument("--dist", type=str, default="dot", help="[dot | l2 | text_cls | text_span | text_marker]")
    parser.add_argument("--support_reduction", type=str, default="score", help="[score | prob]")
    parser.add_argument("--scoring_func", type=str, default=None, help="[activation | bilinear]")
    args = parser.parse_args()

    if args.resume_from_ckpt is None:
        main(args)
    else:
        resume(args)