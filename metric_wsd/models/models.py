import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from metric_wsd.models.context_encoder import ContextEncoder, PairContextEncoder
from metric_wsd.config import Config

config = Config()


class CBERTLinear(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.wn_senses = args.wn_senses
        self.training_words = args.training_words
        self.context_encoder = ContextEncoder(args)

        self.all_senses = [sensekey for _, sensekeys in self.wn_senses.items() for sensekey in sensekeys]
        self.sensekey_to_id = {sensekey: i for i, sensekey in enumerate(self.all_senses)}
        self.sensekey_linear = nn.Linear(config.BERT_DIM, len(self.all_senses))
        
    def forward(self, batch, device):
        context_ids = batch['context_ids']
        spans = batch['context_spans']
        targetwords = batch['context_targetwords']
        examplekeys = batch['context_examplekeys']
        target_ids = batch['target_ids']

        reps = self.context_encoder(context_ids, spans)
        batch_loss = None
        correct = []
        for word, rep, target_id in zip(targetwords, reps, target_ids):
            sense_ids = [self.sensekey_to_id[sensekey] for sensekey in self.wn_senses[word]]
            linear_weights = self.sensekey_linear.weight[sense_ids, :]
            linear_biases = self.sensekey_linear.bias[sense_ids]
            out = linear_weights @ rep + linear_biases
            pred = torch.argmax(out)
            correct.append(pred == target_id)

            loss = F.cross_entropy(out.unsqueeze(0), target_id.unsqueeze(0))
            if batch_loss is None:
                batch_loss = loss
            else:
                batch_loss += loss

        return batch_loss / len(targetwords), torch.stack(correct).detach()
    
    def forward_eval(self, batch, device):
        context_ids = batch['context_ids']
        spans = batch['context_spans']
        targetwords = batch['context_targetwords']
        examplekeys = batch['context_examplekeys']
        target_ids = batch['target_ids']

        with torch.no_grad():
            reps = self.context_encoder(context_ids, spans)
            correct = []
            preds = []
            for word, rep, target_id, examplekey in zip(targetwords, reps, target_ids, examplekeys):
                target_id = target_id.item()
                candidate_sensekeys = self.wn_senses[word]
                if word in self.training_words:
                    sense_ids = [self.sensekey_to_id[sensekey] for sensekey in candidate_sensekeys]
                    linear_weights = self.sensekey_linear.weight[sense_ids, :]
                    linear_biases = self.sensekey_linear.bias[sense_ids]
                    out = linear_weights @ rep + linear_biases
                    pred_idx = torch.argmax(out).item()
                else:
                    pred_idx = 0
                
                preds.append(f'{examplekey} {candidate_sensekeys[pred_idx]}')
                correct.append(pred_idx == target_id)
            return correct, preds

    def forward_eval_return_rep(self, batch, device):
        assert len(batch['context_targetwords']) == 1

        context_ids = batch['context_ids']
        spans = batch['context_spans']
        targetword = batch['context_targetwords'][0]
        examplekey = batch['context_examplekeys'][0]
        sensekey = batch['context_sensekeys'][0]
        target_ids = batch['target_ids']
        
        with torch.no_grad():
            query_rep = self.context_encoder(context_ids, spans)
            return query_rep, sensekey, targetword


class CBERTProto(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.wn_senses = args.wn_senses
        self.training_words = args.training_words
        self.context_encoder = ContextEncoder(args)
    
    def forward(self, batch, device):
        target_ids = batch['target_ids']
        support_ids = batch['support_ids']
        support_spans = batch['support_spans']
        query_ids = batch['query_ids']
        query_spans = batch['query_spans']

        support_reps = []
        for s_ids, s_spans in zip(support_ids, support_spans):
            s_ids = torch.tensor(s_ids).to(device)
            support_rep = self.context_encoder(s_ids, s_spans).mean(dim=0)
            support_reps.append(support_rep)
        support_reps = torch.stack(support_reps)
        query_reps = self.context_encoder(query_ids, query_spans)
        if self.args.dist == 'dot':
            scores = query_reps @ support_reps.t()
        elif self.args.dist == 'l2':
            scores = -torch.cdist(query_reps, support_reps).pow(2)
            
        preds = torch.argmax(scores, dim=1)
        loss = F.cross_entropy(scores, target_ids)
        return loss, preds == target_ids

    def forward_eval(self, batch, device):
        with torch.no_grad():
            support_ids = batch['support_ids']
            support_spans = batch['support_spans']

            query_ids = batch['query_ids']
            query_spans = batch['query_spans']
            sensekey = batch['sensekey']
            targetword = batch['targetword']

            examplekey = batch['examplekey']
            support_sensekeys = batch['support_sensekeys']

            correct = []
            preds = []
            candidate_sensekeys = self.wn_senses[targetword]
            target_id = candidate_sensekeys.index(sensekey)

            if not support_ids:
                # word unseen in training data
                pred_idx = 0
            else:
                query_rep = self.context_encoder(query_ids, query_spans)

                support_reps = []
                for s_ids, s_spans in zip(support_ids, support_spans):
                    s_rep = self.context_encoder(s_ids, s_spans).mean(dim=0)
                    support_reps.append(s_rep)
                support_reps = torch.stack(support_reps)
                if self.args.dist == 'dot':
                    scores = (support_reps * query_rep).mean(dim=1)
                elif self.args.dist == 'l2':
                    scores = -(support_reps - query_rep).pow(2).sum(dim=-1)
                    

                pred_support_idx = torch.argmax(scores).item()
                pred_sensekey = support_sensekeys[pred_support_idx]
                pred_idx = candidate_sensekeys.index(pred_sensekey)
            preds.append(f'{examplekey} {candidate_sensekeys[pred_idx]}')
            correct.append(pred_idx == target_id)
            return correct, preds

    def forward_eval_return_rep(self, batch, device):
        support_ids = batch['support_ids']
        support_spans = batch['support_spans']

        query_ids = batch['query_ids']
        query_spans = batch['query_spans']
        sensekey = batch['sensekey']
        targetword = batch['targetword']

        examplekey = batch['examplekey']
        support_sensekeys = batch['support_sensekeys']

        candidate_sensekeys = self.wn_senses[targetword]
        target_id = candidate_sensekeys.index(sensekey)

        with torch.no_grad():
            query_rep = self.context_encoder(query_ids, query_spans)
            return query_rep, sensekey, targetword


class CBERTProtoConcat(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.wn_senses = args.wn_senses
        self.training_words = args.training_words
        self.context_encoder = ContextEncoder(args)
        self.linear = nn.Linear(config.BERT_DIM * 2, config.BERT_DIM * 2)
        self.reduce_linear = nn.Linear(config.BERT_DIM * 2, 1)
    
    def aggregate_scores_by_label(self, scores, labels, support_reduction, device):
        """
        Given scores tensor([[1.0, 2.0, 3.0]]) and labels tensor([[0, 0, 1]]),
        the function will average the scores for the same labels.
        New scores become tensor([[1.5, 3.0]]) for labels tensor([[0, 1]]).

        Alternatively, `labels` can also be a list of sensekeys. In that case,
        a new label tensor will be generated.

        If `support_reduction` = 'score', then the aggregated scores will be averaged.
        If `support_reduction` = 'prob', then the aggregated scores will be summed.

        """
        if isinstance(labels, list):
            label_to_id = {}
            for label in labels:
                if label not in label_to_id:
                    label_to_id[label] = len(label_to_id)
            labels = torch.tensor([label_to_id[l] for l in labels]).long()
            id_to_label = {v: k for k, v in label_to_id.items()}
        else:
            labels = labels[0]
            id_to_label = None
        M = torch.zeros(labels.max() + 1, labels.size(0)).to(device)
        M[labels, torch.arange(labels.size(0))] = 1.0
        if support_reduction == 'score':
            M = F.normalize(M, p=1, dim=1)
        return scores @ M.t(), id_to_label
    
    def forward(self, batch, device):
        query_target_ids = batch['target_ids']
        support_ids = batch['support_ids']
        support_spans = batch['support_spans']
        query_ids = batch['query_ids']
        query_spans = batch['query_spans']

        support_reps = []
        support_target_ids = []
        for i, (s_ids, s_spans) in enumerate(zip(support_ids, support_spans)):
            s_ids = torch.tensor(s_ids).to(device)
            support_rep = self.context_encoder(s_ids, s_spans)
            support_reps.append(support_rep)
            support_target_ids += [i] * s_ids.size(0)
        support_reps = torch.cat(support_reps, dim=0)
        query_reps = self.context_encoder(query_ids, query_spans)

        num_supports = support_reps.size(0)
        num_queries = query_reps.size(0)

        query_reps = torch.repeat_interleave(query_reps, num_supports, dim=0)
        support_reps = support_reps.repeat(num_queries, 1)
        concat_reps = torch.cat([query_reps, support_reps], dim=1)
        scores = self.reduce_linear(F.relu(self.linear(concat_reps))).view(num_queries, num_supports)

        if self.args.support_reduction == 'score':
            scores, _ = self.aggregate_scores_by_label(scores, support_target_ids, self.args.support_reduction, device)
            preds = torch.argmax(scores, dim=1)
            loss = F.cross_entropy(scores, query_target_ids)
        elif self.args.support_reduction == 'prob':
            probs = F.softmax(scores, dim=1)
            probs, _ = self.aggregate_scores_by_label(probs, support_target_ids, self.args.support_reduction, device)

            preds = torch.argmax(probs, dim=1)
            loss = F.nll_loss(torch.log(probs), query_target_ids)
        return loss, preds == query_target_ids

    def forward_eval(self, batch, device):
        with torch.no_grad():
            support_ids = batch['support_ids']
            support_spans = batch['support_spans']

            query_ids = batch['query_ids']
            query_spans = batch['query_spans']
            sensekey = batch['sensekey']
            targetword = batch['targetword']

            examplekey = batch['examplekey']
            support_sensekeys = batch['support_sensekeys']

            correct = []
            preds = []
            candidate_sensekeys = self.wn_senses[targetword]
            target_id = candidate_sensekeys.index(sensekey)

            if not support_ids:
                # word unseen in training data
                pred_idx = 0
            else:
                query_rep = self.context_encoder(query_ids, query_spans)

                support_reps = []
                for s_ids, s_spans in zip(support_ids, support_spans):
                    s_rep = self.context_encoder(s_ids, s_spans)
                    support_reps.append(s_rep)
                support_reps = torch.cat(support_reps, dim=0)

                num_supports = support_reps.size(0)

                query_reps = query_rep.repeat(num_supports, 1)
                concat_reps = torch.cat([query_reps, support_reps], dim=1)

                scores = self.reduce_linear(F.relu(self.linear(concat_reps))).reshape(1, num_supports)

                if self.args.support_reduction == 'score':
                    scores, id_to_label = self.aggregate_scores_by_label(scores, support_sensekeys, self.args.support_reduction, device)
                elif self.args.support_reduction == 'prob':
                    probs = F.softmax(scores, dim=1)
                    probs, id_to_label = self.aggregate_scores_by_label(probs, support_sensekeys, self.args.support_reduction, device)
                    scores = probs

                pred_support_idx = torch.argmax(scores).item()
                pred_sensekey = id_to_label[pred_support_idx]
                pred_idx = candidate_sensekeys.index(pred_sensekey)
            preds.append(f'{examplekey} {candidate_sensekeys[pred_idx]}')
            correct.append(pred_idx == target_id)
            return correct, preds


class PairCBERTProto(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.wn_senses = args.wn_senses
        self.training_words = args.training_words
        self.pair_context_encoder = PairContextEncoder(args)
    
    def aggregate_scores_by_label(self, scores, labels, support_reduction, device):
        """
        Given scores tensor([[1.0, 2.0, 3.0]]) and labels tensor([[0, 0, 1]]),
        the function will average the scores for the same labels.
        New scores become tensor([[1.5, 3.0]]) for labels tensor([[0, 1]]).

        Alternatively, `labels` can also be a list of sensekeys. In that case,
        a new label tensor will be generated.

        If `support_reduction` = 'score', then the aggregated scores will be averaged.
        If `support_reduction` = 'prob', then the aggregated scores will be summed.

        """
        if isinstance(labels, list):
            label_to_id = {}
            for label in labels:
                if label not in label_to_id:
                    label_to_id[label] = len(label_to_id)
            labels = torch.tensor([label_to_id[l] for l in labels]).long()
            id_to_label = {v: k for k, v in label_to_id.items()}
        else:
            labels = labels[0]
            id_to_label = None

        M = torch.zeros(labels.max() + 1, labels.size(0)).to(device)
        M[labels, torch.arange(labels.size(0))] = 1.0
        if support_reduction == 'score':
            M = F.normalize(M, p=1, dim=1)
        return scores @ M.t(), id_to_label
    
    def forward(self, batch, device):
        paired_ids = batch['paired_ids']
        paired_spans = batch['paired_spans']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        query_target_ids = batch['query_target_ids']
        support_target_ids = batch['support_target_ids']

        scores = self.pair_context_encoder(paired_ids, token_type_ids, attention_mask, paired_spans)
        if self.args.support_reduction == 'score':
            scores, _ = self.aggregate_scores_by_label(scores, support_target_ids, self.args.support_reduction, device)
            preds = torch.argmax(scores, dim=1)
            loss = F.cross_entropy(scores, query_target_ids)
        elif self.args.support_reduction == 'prob':
            probs = F.softmax(scores, dim=1)
            probs, _ = self.aggregate_scores_by_label(probs, support_target_ids, self.args.support_reduction, device)

            preds = torch.argmax(probs, dim=1)
            loss = F.nll_loss(torch.log(probs), query_target_ids)

        return loss, preds == query_target_ids

    def forward_eval(self, batch, device):
        with torch.no_grad():
            paired_ids = batch['paired_ids']
            token_type_ids = batch['token_type_ids']
            attention_mask = batch['attention_mask']
            paired_spans = batch['paired_spans']

            support_sensekeys = batch['support_sensekeys']
            sensekey = batch['sensekey']
            examplekey = batch['examplekey']
            targetword = batch['targetword']

            correct = []
            preds = []
            candidate_sensekeys = self.wn_senses[targetword]
            target_id = candidate_sensekeys.index(sensekey)

            if not support_sensekeys:
                # word unseen in training data
                pred_idx = 0
            else:
                scores = self.pair_context_encoder(paired_ids, token_type_ids, attention_mask, paired_spans)
                if self.args.support_reduction == 'score':
                    scores, id_to_label = self.aggregate_scores_by_label(scores, support_sensekeys, self.args.support_reduction, device)
                elif self.args.support_reduction == 'prob':
                    probs = F.softmax(scores, dim=1)
                    probs, id_to_label = self.aggregate_scores_by_label(probs, support_sensekeys, self.args.support_reduction, device)
                    scores = probs
                pred_support_idx = torch.argmax(scores).item()
                pred_sensekey = id_to_label[pred_support_idx]
                pred_idx = candidate_sensekeys.index(pred_sensekey)
            preds.append(f'{examplekey} {candidate_sensekeys[pred_idx]}')
            correct.append(pred_idx == target_id)
            return correct, preds