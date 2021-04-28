import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizerFast, BertModel, BertConfig

from metric_wsd.config import Config

config = Config()


def get_subtoken_indecies(text_tokens, offset_mapping, offset):
    if offset == 0:
        start = 0
        end = len(text_tokens[offset])
    else:
        start = len(' '.join(text_tokens[:offset])) + 1
        end = start + len(text_tokens[offset])

    start_token_pos = 0
    end_token_pos = 0
    for i, (s, e) in enumerate(offset_mapping):
        if start == s:
            start_token_pos = i
        if end == e:
            end_token_pos = i + 1
    if offset == 0:
        start_token_pos = 1
    return start_token_pos, end_token_pos


def extend_span_offset(paired_spans, offset_mappings):
    """
    Extend span offset for query.
    TODO: Document it better. This function is error prone.
    """
    paired_spans_ = []
    for paired_span, offset_mapping in zip(paired_spans, offset_mappings):
        loc = []  # indices for [CLS], first [SEP], and second [SEP]
        for i, (s, e) in enumerate(offset_mapping):
            if s == 0 and e == 0:
                loc.append(i)
        s_paired_span = list(paired_span[0])
        q_paired_span = [paired_span[1][0] + loc[1], paired_span[1][1] + loc[1]]
        paired_spans_.append([s_paired_span, q_paired_span])
    
    return torch.tensor(paired_spans_).long()


class ContextEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

        if not args.freeze_context_enc:
            self.model.train()
        else:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
    
    def forward(self, input_ids, spans):
        output_tensor = self.model(input_ids)[0]
        outputs = []
        for span, output_tensor_slice in zip(spans, output_tensor):
            outputs.append(output_tensor_slice[span[0]:span[1], :].mean(dim=0))
        return torch.stack(outputs)

    def encode(self, text_tokens, offsets):
        """Take in a single text string  Return representations with subtokens merged."""
        reps = []
        encoded = self.tokenizer.encode_plus(' '.join(text_tokens), return_tensors='pt', return_offsets_mapping=True)

        with torch.no_grad():
            output_tensor = self.model(encoded['input_ids'])[0]

        for offset in offsets:
            start, end = get_subtoken_indecies(text_tokens, encoded['offset_mapping'], offset)
            reps.append(output_tensor[:, start:end, :].mean(dim=1))
        return reps


class PairContextEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

        if self.args.dist == 'text_cls':
            self.linear = nn.Linear(config.BERT_DIM, 1)

        elif self.args.dist == 'text_span' and self.args.scoring_func == 'activation':
            self.linear = nn.Linear(config.BERT_DIM * 2, 1)
            self.proj = nn.Linear(config.BERT_DIM * 2, config.BERT_DIM * 2)

        elif self.args.dist == 'text_span' and self.args.scoring_func == 'bilinear':
            self.bilinear = nn.Linear(config.BERT_DIM, config.BERT_DIM)

        else:
            self.linear = nn.Linear(config.BERT_DIM * 2, 1)

        if self.args.dist == 'text_marker':
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['[LMRK]', '[RMRK]']})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, paired_ids, token_type_ids, attention_mask, paired_spans):
        """
        paired_ids: num_queries x num_supports x max_length
        token_type_ids: num_queries x num_supports x max_length
        attention_mask: num_queries x num_supports x max_length
        paired_spans: num_queries x num_supports x 2 (support_span & query_span) x 2 (start & end)
        """
        num_queries, num_supports, max_length = paired_ids.size()

        paired_ids = paired_ids.reshape(-1, max_length)
        token_type_ids = token_type_ids.reshape(-1, max_length)
        attention_mask = attention_mask.reshape(-1, max_length)
        paired_spans = paired_spans.reshape(-1, 2, 2)

        output_tensor = self.model(paired_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

        outputs = []
        for i, (paired_span, output_tensor_slice) in enumerate(zip(paired_spans, output_tensor)):
            query_span = paired_span[0, :]
            support_span = paired_span[1, :]

            if self.args.dist == 'text_cls':
                rep = output_tensor_slice[0, :]  # take [CLS]
            elif self.args.dist == 'text_span':
                query_rep = output_tensor_slice[query_span, :].mean(dim=0)
                support_rep = output_tensor_slice[support_span, :].mean(dim=0)
                rep = torch.cat([query_rep, support_rep], dim=0)
            elif self.args.dist == 'text_marker':
                # use [LMRK] for both query and support
                query_rep = output_tensor_slice[query_span[0] - 1, :]
                support_rep = output_tensor_slice[support_span[0] - 1, :]
                rep = torch.cat([query_rep, support_rep], dim=0)
            outputs.append(rep)
        outputs = torch.stack(outputs).view(num_queries, num_supports, -1)

        if self.args.scoring_func == 'activation':
            outputs = self.linear(F.relu(self.proj(outputs)))
        elif self.args.scoring_func == 'bilinear':
            q = outputs[:, :, :config.BERT_DIM]
            s = outputs[:, :, config.BERT_DIM:]
            outputs = (q * self.bilinear(s)).mean(dim=-1, keepdim=True)
        elif self.args.scoring_func == 'dot':
            q = outputs[:, :, :config.BERT_DIM]
            s = outputs[:, :, config.BERT_DIM:]
            outputs = (q * s).mean(dim=-1, keepdim=True)
        else:
            outputs = self.linear(outputs)
        return outputs.squeeze(-1)


if __name__ == '__main__':
    context_encoder = ContextEncoder(args=None)
    context_encoder.encode('this is a system that is very complicated', [3])