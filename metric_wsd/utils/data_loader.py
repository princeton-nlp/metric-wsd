import re
import json
import random
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import BertTokenizerFast
import pytorch_lightning as pl

from metric_wsd.models.context_encoder import get_subtoken_indecies, extend_span_offset
from metric_wsd.utils.utils import sample_examples, load_glosses
from metric_wsd.config import Config
from metric_wsd.baselines.wsd_biencoders.wsd_models.util import load_data, load_wn_senses

config = Config()


class WSDDataLoader:
    def __init__(self, args, use_train_data_as_dev=False):
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.tokenizer_kwargs = {
            'max_length': self.args.max_length,
            'padding': 'max_length',
            'truncation': True,
            'return_offsets_mapping': True,
            'return_tensors': 'pt',
        }
        self.tokenizer_gloss_kwargs = {
            'max_length': 64,
            'padding': 'max_length',
            'truncation': True,
            'return_offsets_mapping': True,
            'return_tensors': 'pt',
        }
        if self.args.dist == 'text_marker':
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['[LMRK]', '[RMRK]']})

        self.wn_senses = load_wn_senses(config.WN)
        self.sensekey_to_gloss = load_glosses(config.GLOSS)
        self.task_to_classes = None
        self.task_freq = None
        self.sense_freq = None
        self.use_train_data_as_dev = use_train_data_as_dev
        self._dataloaders = {}
        for mode in ['train', 'dev', 'test']:
            data = self._load_raw_data(mode)
            self._build_dataloader(data, mode)

    def _load_raw_data(self, mode):
        data_ = []
        if mode == 'train':
            data = load_data(*config.SEMCOR)

        elif mode == 'dev':
            if self.use_train_data_as_dev:
                data = load_data(*config.SEMCOR)
            else:
                data = load_data(*config.SE07)
        elif mode == 'test':
            data = load_data(*config.ALL)
        
        if mode == 'train':
            if self.args.debug:
                data = data[:100]
        for sent in data:
            original_text_tokens = list(map(list, zip(*sent)))[0]

            for offset, (_, stem, pos, examplekey, sensekey) in enumerate(sent):
                if sensekey != -1:
                    if self.args.dist == 'text_marker':
                        text_tokens = original_text_tokens[:offset] + \
                                      ['[LMRK]', original_text_tokens[offset], '[RMRK]'] + \
                                      original_text_tokens[offset + 1:]
                        offset += 1
                    else:
                        text_tokens = original_text_tokens[:]

                    encoded = self.tokenizer.encode_plus(' '.join(text_tokens), **self.tokenizer_kwargs)

                    start, end = get_subtoken_indecies(text_tokens, encoded['offset_mapping'][0].tolist(), offset)
                    if start >= end:
                        continue
                    if self.args.dist.startswith('text') and end >= (self.args.max_length // 2 - 1):
                        # cut the original text tokens
                        half = 20
                        assert offset > half
                        new_text_tokens = text_tokens[offset - half:offset + half + 1]
                        new_encoded = self.tokenizer.encode_plus(' '.join(new_text_tokens), **self.tokenizer_kwargs)
                        new_start, new_end = get_subtoken_indecies(new_text_tokens, new_encoded['offset_mapping'][0].tolist(), half)
                        if new_end >= (self.args.max_length // 2 - 1):
                            continue
                        data_.append((new_text_tokens, stem + '+' + config.pos_map[pos], examplekey, sensekey, half, [new_start, new_end]))
                    else:
                        data_.append((text_tokens, stem + '+' + config.pos_map[pos], examplekey, sensekey, offset, [start, end]))

        if mode == 'train':
            self.task_to_classes = {}
            for text_tokens, targetword, examplekey, sensekey, offset, span in data_:
                example = (text_tokens, targetword, examplekey, sensekey, offset, span)

                if targetword not in self.task_to_classes:
                    self.task_to_classes[targetword] = OrderedDict({sensekey: [example]})
                elif targetword in self.task_to_classes and sensekey not in self.task_to_classes[targetword]:
                    self.task_to_classes[targetword][sensekey] = [example]
                elif targetword in self.task_to_classes and sensekey in self.task_to_classes[targetword]:
                    self.task_to_classes[targetword][sensekey].append(example)
                else:
                    raise ValueError('Should not happen.')
            self.training_words = set([word for word in self.task_to_classes.keys()])
            self.task_freq = {task: sum([len(examples) for examples in sense_to_examples.values()]) 
                              for task, sense_to_examples in self.task_to_classes.items()}
            
            self.sense_freq = defaultdict(int)
            for task, sense_to_examples in self.task_to_classes.items():
                for sensekey, examples in sense_to_examples.items():
                    self.sense_freq[sensekey] += len(examples)
        return data_

    def _build_dataloader(self, data, mode):
        if not self.args.episodic:
            dataset = WSDDataset(
                data, self.args,
                self.tokenizer, self.tokenizer_kwargs, self.tokenizer_gloss_kwargs,
                self.wn_senses, self.sensekey_to_gloss, self.task_freq
            )
            collate_fn = dataset.collater
        else:
            dataset = WSDEpisodicDataset(
                self.task_to_classes, self.args,
                self.tokenizer, self.tokenizer_kwargs, self.tokenizer_gloss_kwargs,
                self.wn_senses, mode, data, self.sensekey_to_gloss, self.task_freq
            )
            collate_fn = dataset.collater if mode == 'train' else dataset.eval_collator
        
        shuffle = not self.args.no_shuffle if mode == 'train' and not self.args.episodic else False

        self._dataloaders[mode] = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            worker_init_fn=np.random.seed(0),
            num_workers=0,
            collate_fn=collate_fn
        )
        print(f'[{mode}] dataloader (iterator) built.')
    
    @property
    def train(self):
        return self._dataloaders['train']

    @property
    def dev(self):
        return self._dataloaders['dev']
    
    @property
    def test(self):
        return self._dataloaders['test']


class WSDDataset(Dataset):
    def __init__(self,
                 data,
                 args,
                 tokenizer,
                 tokenizer_kwargs,
                 tokenizer_gloss_kwargs,
                 wn_senses,
                 sensekey_to_gloss,
                 task_freq):
        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.tokenizer_gloss_kwargs = tokenizer_gloss_kwargs
        self.wn_senses = wn_senses
        self.sensekey_to_gloss = sensekey_to_gloss
        self.task_freq = task_freq
    
    def __getitem__(self, index):
        # batch: text_tokens, targetword, examplekey, sensekey, offset, span
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def collater(self, batch):
        texts, targetwords, examplekeys, sensekeys, offsets, spans = list(map(list, zip(*batch)))

        target_ids = torch.tensor([
            self.wn_senses[targetword].index(sensekey)
            for targetword, sensekey in zip(targetwords, sensekeys)
        ]).long()
        
        encoded = self.tokenizer.batch_encode_plus(
            [' '.join(text) for text in texts],
            max_length=self.args.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt',
        )

        return {
            'context_ids': encoded['input_ids'] if self.args.gpus == 0 else encoded['input_ids'].cuda(),
            'context_texts': texts,
            'context_targetwords': targetwords,
            'context_examplekeys': examplekeys,
            'context_sensekeys': sensekeys,
            'context_offsets': offsets,
            'context_spans': spans,
            'target_ids': target_ids if self.args.gpus == 0 else target_ids.cuda(),
            'batch_size': target_ids.size(0)
        }


class WSDEpisodicDataset(IterableDataset):
    def __init__(self,
                 task_to_classes,
                 args,
                 tokenizer,
                 tokenizer_kwargs,
                 tokenizer_gloss_kwargs,
                 wn_senses,
                 mode,
                 data,
                 sensekey_to_gloss,
                 task_freq):
        self.args = args
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.tokenizer_gloss_kwargs = tokenizer_gloss_kwargs
        self.wn_senses = wn_senses
        self.task_to_classes = task_to_classes
        self.mode = mode
        self.data = data
        self.seed = 0
        self.sensekey_to_gloss = sensekey_to_gloss
        self.task_freq = task_freq
    
    def __iter__(self):
        if self.mode == 'train':
            return self.get_episode()
        else:
            return self.get_eval_data(self.data)

    def get_episode(self):
        random.seed(self.args.seed)
        all_tasks = list(self.task_to_classes.keys())
        random.shuffle(all_tasks)
        for i_episode in range(len(all_tasks)):
            task = all_tasks[i_episode]
            freq = self.task_freq[task]
            class_to_examples = self.task_to_classes[task]
            selected_examples = {}

            if self.args.mix_strategy:
                # MixStrategy - uses all the examples and ignore "balanced" or "uniform" sampling
                sensekey_to_num_sampled = {sensekey: len(examples) for sensekey, examples in class_to_examples.items() if len(examples) != 0}
            elif freq >= self.args.sample_strategy_threshold:
                # MaxQuery
                sense_candidates = [sensekey for sensekey, examples in class_to_examples.items() if len(examples) != 0]
                random.shuffle(sense_candidates)
                sense_candidates = set(sense_candidates[:self.args.ks_support_kq_query[2]])
                sensekey_to_num_sampled = {sensekey: len(examples) for sensekey, examples in class_to_examples.items() if sensekey in sense_candidates}
            elif freq < self.args.sample_strategy_threshold:
                # RatioSplit - "balanced" or "uniform" only works in this case
                sensekeys, weights = zip(*[(sensekey, len(examples)) for sensekey, examples in class_to_examples.items()])
                if self.args.sampling == 'balanced':
                    weights = [1 for _ in range(len(sensekeys))]
                sensekey_to_num_sampled = sample_examples(sensekeys, weights, k=self.args.episodic_k, seed=self.args.seed + i_episode)
            else:
                raise ValueError('Should not happen.')

            for sensekey, num in sensekey_to_num_sampled.items():
                examples = class_to_examples[sensekey]
                random.shuffle(examples)
                selected = examples[:num]  # num > len(examples) could happend. 
                selected_examples[sensekey] = selected

            if len(selected_examples) <= 1:
                continue
            # if every sense only has one example then there's no way to 
            # separate them into meaningful support/query split
            if sum([len(examples) for examples in selected_examples.values()]) == len(selected_examples):
                continue

            batches = self.split_examples_to_supports_and_queries(task, selected_examples)
            for batch in batches:
                yield batch
        self.args.seed += 1

    def split_examples_to_supports_and_queries(self, task, selected_examples: dict):
        """
        Given a task (word), split its selected training examples into a query set and a support set.
        In the case of mix_strategy, all training examples are selected.
        Three cases:
            1. `mix_strategy` & 50 <= freq or freq > 400:
                Split by `support_query_ratio` -> supports & queries. If the number of supports is too large (> episodic_k),
                then use `avg_num_supports` for each support per sense instead.

            2. `mix_strategy` & 50 < freq <= 400:
                Assign `max_supports` to supports for each sense key and the rest to queries.
            
            3. If task freq >= `sample_strategy_threshold`:
                The MaxQuery variant.
                Assign `max_supports` to supports for each sense key and the rest to queries.
            
            4. If task freq < `sample_strategy_threshold`:
                The RatioSplit variant.
                Split by `support_query_ratio`.
        """
        supports = defaultdict(list)
        queries = defaultdict(list)

        freq = self.task_freq[task]

        if self.args.mix_strategy and (freq <= 50 or freq > 400):
            all_examples = [example for examples in selected_examples.values() for example in examples]
            random.shuffle(all_examples)
            num_supports_to_sample = int(len(all_examples) * self.args.support_query_ratio)
            rest = []
            for example in all_examples:
                _, _, _, sensekey, _, _ = example
                if not supports[sensekey]:
                    # make sure the support set has at least one example per sense
                    supports[sensekey].append(example)
                else:
                    rest.append(example)
            cutoff = num_supports_to_sample - len(supports)
            for i, example in enumerate(rest):
                _, _, _, sensekey, _, _ = example
                if i < cutoff:
                    supports[sensekey].append(example)
                else:
                    queries[sensekey].append(example)

            sensekeys = [_ for _ in supports.keys()]
            if num_supports_to_sample > self.args.episodic_k:
                avg_num_supports = self.args.avg_num_supports
                evened_supports = {}
                for sensekey, examples in supports.items():
                    if len(examples) > avg_num_supports:
                        evened_supports[sensekey] = examples[:avg_num_supports]
                        queries[sensekey] += examples[avg_num_supports:]
                    else:
                        evened_supports[sensekey] = examples
                supports = evened_supports

            random.shuffle(sensekeys)
            sensekeys = sensekeys[:self.args.max_senses]
            sensekey_set = set(sensekeys)
            supports = {sensekey: examples for sensekey, examples in supports.items() if sensekey in sensekey_set}
            queries = {sensekey: examples for sensekey, examples in queries.items() if sensekey in sensekey_set}

        elif self.args.mix_strategy and (50 < freq <= 400):
            # sampling strategy 1: max query
            max_supports, max_queries = self.args.ks_support_kq_query[:2]
            for sensekey, examples in selected_examples.items():
                supports[sensekey] += examples[:max_supports]
                #q_examples = examples[max_supports:max_supports + max_queries]
                q_examples = examples[max_supports:]
                if not q_examples and len(supports[sensekey]) > 1:
                    queries[sensekey].append(supports[sensekey].pop())
                else:
                    queries[sensekey] += q_examples
            queries = {sensekey: examples for sensekey, examples in queries.items() if len(examples) != 0}  # remove empty sensekeys

        elif freq >= self.args.sample_strategy_threshold:
            # sampling strategy 1: MaxQuery
            max_supports, max_queries = self.args.ks_support_kq_query[:2]
            for sensekey, examples in selected_examples.items():
                supports[sensekey] += examples[:max_supports]
                q_examples = examples[max_supports:]
                if not q_examples and len(supports[sensekey]) > 1:
                    queries[sensekey].append(supports[sensekey].pop())
                else:
                    queries[sensekey] += q_examples
            queries = {sensekey: examples for sensekey, examples in queries.items() if len(examples) != 0}  # remove empty sensekeys
    
        elif freq < self.args.sample_strategy_threshold:
            # sampling stategy 2: RatioSplit
            selected_examples = [example for examples in selected_examples.values() for example in examples]
            random.shuffle(selected_examples)
            rest = []
            for example in selected_examples:
                _, _, _, sensekey, _, _ = example
                if not supports[sensekey]:
                    # make sure the support set has at least one example per sense
                    supports[sensekey].append(example)
                else:
                    rest.append(example)
            cutoff = int(len(rest) * self.args.support_query_ratio)
            for i, example in enumerate(rest):
                _, _, _, sensekey, _, _ = example
                if i < cutoff:
                    supports[sensekey].append(example)
                else:
                    queries[sensekey].append(example)
        else:
            raise ValueError('Should not happen.')

        num_supports = sum([len(_) for _ in supports.values()])
        num_queries = sum([len(_) for _ in queries.values()])

        sensekeys = [_ for _ in supports.keys()]
        class_to_examples = self.task_to_classes[task]

        queries = [example for sensekey, examples in queries.items() for example in examples]

        batches = []
        for i in range(len(queries) // self.args.max_queries + 1):
            sub_queries = queries[i * self.args.max_queries:(i + 1) * self.args.max_queries]
            if sub_queries:
                batches.append([supports, sub_queries])

        return batches

    def create_text_pair_batch(self, supports, queries):
        """
        supports: Dict[str, List[Examples]]
        queries: List[Examples]
        """
        sensekey_to_id = {}

        for i, (sensekey, examples) in enumerate(supports.items()):
            sensekey_to_id[sensekey] = i

        supports = [example for examples in supports.values() for example in examples]

        support_target_ids = []
        for s_example in supports:
            s_text, s_targetword, s_examplekey, s_sensekey, s_offset, s_span = s_example
            support_target_ids.append(sensekey_to_id[s_sensekey])

        query_target_ids = []
        for q_example in queries:
            q_text, q_targetword, q_examplekey, q_sensekey, q_offset, q_span = q_example
            query_target_ids.append(sensekey_to_id[q_sensekey])

        paired_texts = []
        paired_spans = []
        for i, q_example in enumerate(queries):
            for j, s_example in enumerate(supports):
                q_text, q_targetword, q_examplekey, q_sensekey, q_offset, q_span = q_example
                s_text, s_targetword, s_examplekey, s_sensekey, s_offset, s_span = s_example

                paired_texts.append((' '.join(q_text), ' '.join(s_text)))
                paired_spans.append((q_span, s_span))

        encoded = self.tokenizer.batch_encode_plus(paired_texts, **self.tokenizer_kwargs)

        # dimensions: num_queries x num_supports x max_length
        paired_ids = encoded['input_ids'].view(len(queries), len(supports), -1)
        token_type_ids = encoded['token_type_ids'].view(len(queries), len(supports), -1)
        attention_mask = encoded['attention_mask'].view(len(queries), len(supports), -1)
        paired_spans = extend_span_offset(paired_spans, encoded['offset_mapping'])
        paired_spans = paired_spans.view(len(queries), len(supports), 2, 2)

        query_target_ids = torch.tensor(query_target_ids).long()
        support_target_ids = torch.tensor(support_target_ids).long().repeat(query_target_ids.size(0), 1)

        return {
            'paired_ids': paired_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'paired_spans': paired_spans,
            'query_target_ids': query_target_ids,
            'support_target_ids': support_target_ids,
            'batch_size': 1
        }
    
    def create_dual_rep_batch(self, supports, queries):
        """
        supports: Dict[str, List[Example]]
        queries: List[Example]
        """
        support_ids = []
        support_texts = []
        support_spans = []
        sensekey_to_id = {}
        gloss_texts = []
        for i, (sensekey, examples) in enumerate(supports.items()):
            s_texts, s_targetwords, s_examplekeys, s_sensekeys, s_offsets, s_spans = list(map(list, zip(*examples)))
            s_encoded = self.tokenizer.batch_encode_plus([' '.join(text) for text in s_texts], **self.tokenizer_kwargs)
            support_ids.append(s_encoded['input_ids'])
            support_texts.append(s_texts)
            support_spans.append(s_spans)
            sensekey_to_id[sensekey] = i
            gloss_texts.append(self.sensekey_to_gloss[sensekey])

        gloss_ids = self.tokenizer.batch_encode_plus(gloss_texts, **self.tokenizer_gloss_kwargs)['input_ids']

        query_ids = []
        query_texts = []
        query_spans = []
        target_ids = []
        query_sensekeys = []
        for example in queries:
            q_text, q_targetword, q_examplekey, q_sensekey, q_offset, q_span = example
            query_texts.append(' '.join(q_text))
            query_spans.append(q_span)
            target_ids.append(sensekey_to_id[q_sensekey])
            query_sensekeys.append(q_sensekey)

        q_encoded = self.tokenizer.batch_encode_plus(query_texts, **self.tokenizer_kwargs)

        query_ids = q_encoded['input_ids']
        target_ids = torch.tensor(target_ids).long()

        query_spans = torch.tensor(query_spans).long()
        support_ids = [_.tolist() for _ in support_ids]

        return {
            'query_ids': query_ids,
            'query_spans': query_spans,
            'support_ids': support_ids,
            'support_spans': support_spans,
            'target_ids': target_ids,
            'batch_size': 1,
            'gloss_ids': gloss_ids.repeat(query_ids.size(0), 1, 1),
        }

    def collater(self, batch):
        assert len(batch) == 1
        batch = batch[0]
        supports, queries = batch

        if self.args.dist.startswith('text'):
            return self.create_text_pair_batch(supports, queries)
        else:
            return self.create_dual_rep_batch(supports, queries)

    def get_eval_data(self, data):
        for example in data:
            yield example
    
    def eval_collator(self, batch):
        assert len(batch) == 1
        batch = batch[0]
        text, targetword, examplekey, sensekey, offset, span = batch
        gloss_ids = None
        
        if self.args.dist.startswith('text'):
            paired_ids, token_type_ids, attention_mask, paired_texts, paired_spans, support_sensekeys = [], [], [], [], [], []
        else:
            support_ids, support_spans, support_sensekeys = [], [], []

        if targetword in self.task_to_classes:
            class_to_examples = self.task_to_classes[targetword]
            gloss_ids = []
            freq = self.task_freq[targetword]

            sensekey_to_num_sampled = {sensekey: len(examples) for sensekey, examples in class_to_examples.items()}
            total_examples = sum([len(examples) for examples in class_to_examples.values()])

            for s_key, num in sensekey_to_num_sampled.items():
                examples = class_to_examples[s_key]
                random.shuffle(examples)
                if self.args.max_inference_supports > 0:
                    examples = examples[:self.args.max_inference_supports]
                elif self.args.mix_strategy and (freq <= 50 or freq > 400):
                    num_supports_to_sample = int(total_examples * self.args.support_query_ratio)
                    if num_supports_to_sample > self.args.episodic_k:
                        #avg_num_supports = self.args.episodic_k // self.args.max_senses
                        avg_num_supports = self.args.avg_num_supports
                        examples = examples[:avg_num_supports]
                elif self.args.mix_strategy and (50 < freq <= 400):
                    examples = examples[:self.args.ks_support_kq_query[0]]
                else:
                    raise ValueError('should not happend')
                s_texts, s_targetwords, s_examplekeys, s_sensekeys, s_offsets, s_spans = list(map(list, zip(*examples)))

                if self.args.dist.startswith('text'):
                    paired_texts += [(' '.join(s_text), ' '.join(text)) for s_text in s_texts]
                    paired_spans += [(s_span, span) for s_span in s_spans]
                    support_sensekeys += s_sensekeys
                else:
                    support_encoded = self.tokenizer.batch_encode_plus([' '.join(t) for t in s_texts], **self.tokenizer_kwargs)
                    support_ids.append(support_encoded['input_ids'])
                    support_spans.append(s_spans)
                    if self.args.dist == 'concat':
                        support_sensekeys += s_sensekeys
                    else:
                        support_sensekeys.append(s_sensekeys[0])
                
                gloss_ids.append(self.sensekey_to_gloss[s_key])
            gloss_ids = self.tokenizer.batch_encode_plus(gloss_ids, **self.tokenizer_gloss_kwargs)['input_ids']

        if self.args.dist.startswith('text'):
            if paired_texts:
                encoded = self.tokenizer.batch_encode_plus(paired_texts, **self.tokenizer_kwargs)

                # permute dimensions to num_queries x num_supports x max_length
                paired_ids = encoded['input_ids'].view(len(support_sensekeys), 1, -1).permute(1, 0, 2)
                token_type_ids = encoded['token_type_ids'].view(len(support_sensekeys), 1, -1).permute(1, 0, 2)
                attention_mask = encoded['attention_mask'].view(len(support_sensekeys), 1, -1).permute(1, 0, 2)
                paired_spans = extend_span_offset(paired_spans, encoded['offset_mapping'])
                paired_spans = paired_spans.view(len(support_sensekeys), 1, 2, 2).permute(1, 0, 2, 3)

            return {
                'paired_ids': paired_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'paired_spans': paired_spans,
                'sensekey': sensekey,
                'examplekey': examplekey,
                'support_sensekeys': support_sensekeys,
                'targetword': targetword,
            }
        else:
            encoded = self.tokenizer.encode_plus(' '.join(text), **self.tokenizer_kwargs)
            query_ids = encoded['input_ids'] if self.args.gpus == 0 else encoded['input_ids'].cuda()
            return {
                'query_ids': query_ids,
                'query_spans': [span],
                'sensekey': sensekey,
                'examplekey': examplekey,
                'targetword': targetword,
                'support_ids': support_ids if self.args.gpus == 0 else [ids.cuda() for ids in support_ids], # a list of supports (ids)
                'support_spans': support_spans,
                'support_sensekeys': support_sensekeys,
                'gloss_ids': gloss_ids.repeat(query_ids.size(0), 1, 1) if gloss_ids is not None else gloss_ids
            }