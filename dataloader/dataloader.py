import random

import numpy as np
import torch
import torch.utils.data as data_utils

from .negative_samplers import negative_sampler_factory

from utils import PrintInputShape



class BertDataloader:
    def __init__(self, args, dataset):
        # super().__init__(args, dataset)
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        # seq
        self.train = dataset['tra_item_seq']
        self.val = dataset['val_item_seq']
        self.test = dataset['tes_item_seq']
        # map
        self.umap = dataset['u_i_map']['uid']
        self.smap = dataset['u_i_map']['sid']
        
        self.i2attr_map = dataset['i2attr_map']

        self.user_count = len(self.umap)
        self.item_count = len(self.smap)
        self.attr_count = {attr_name : len(_map) for attr_name, _map in self.i2attr_map.items()}

        args.num_items = len(self.smap)
        args.nums_attrs = {attr_name : len(_map) for attr_name, _map in self.i2attr_map.items()}
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1
        self.ATTRS_MASK_TOKENS = {attr_name : cnt+1 for attr_name, cnt in self.attr_count.items()}
        for attr_name in self.args.attr_names:
            self.i2attr_map[attr_name][self.CLOZE_MASK_TOKEN] = self.ATTRS_MASK_TOKENS[attr_name]
            self.i2attr_map[attr_name][0] = 0


        code = args.train_negative_sampler_code
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed,
                                                          self.save_folder)
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder)

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng,
        self.i2attr_map)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        dataset = BertEvalDataset(self.args, self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples, self.i2attr_map)
        return dataset


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, seq_mask_token, num_items, rng, i2attr_map):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.seq_mask_token = seq_mask_token
        self.num_items = num_items
        self.rng = rng

        self.i2attr_map = i2attr_map

        self.printer = PrintInputShape(3)
        

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]

        tokens = []
        labels = []

        for s, g in zip(seq, gnr):
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.seq_mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        attrs = {}
        for attr_name in self.args.attr_names:
            attrs[attr_name] = torch.LongTensor([self.i2attr_map[attr_name][item] for item in tokens])


        return torch.LongTensor(tokens), torch.LongTensor(labels), attrs



class BertEvalDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2answer, max_len, seq_mask_token, negative_samples, i2attr_map):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.seq_mask_token = seq_mask_token
        self.negative_samples = negative_samples

        self.i2attr_map = i2attr_map


    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)
        seq = self.padding_and_trim(seq, 'seq')

        attrs = {}
        for attr_name in self.i2attr_map.keys():
            attrs[attr_name] = torch.LongTensor([self.i2attr_map[attr_name][item] for item in seq])


        return torch.LongTensor(seq), attrs, torch.LongTensor(candidates), torch.LongTensor(labels)


    def padding_and_trim(self, seq, seq_type):
        if seq_type == 'seq':
            seq = seq + [self.seq_mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        return seq