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
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        # genre
        self.train_g = dataset['train_g']
        self.val_g = dataset['val_g']
        self.test_g = dataset['test_g']
        # map
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.gmap = dataset['gmap']

        self.user_count = len(self.umap)
        self.item_count = len(self.smap)
        self.genre_count = len(self.gmap)

        args.num_items = len(self.smap)
        args.num_genres = len(self.gmap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1
        self.GENRE_MASK_TOKEN = self.genre_count + 1

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
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng, self.train_g,
        self.GENRE_MASK_TOKEN, self.genre_count)
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
        dataset = BertEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples, self.train_g,
            self.GENRE_MASK_TOKEN)
        return dataset


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, seq_mask_token, num_items, rng, u2gnr, gnr_mask_token, num_genres):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.seq_mask_token = seq_mask_token
        self.num_items = num_items
        self.rng = rng

        self.u2gnr = u2gnr

        self.gnr_mask_token = gnr_mask_token

        self.printer = PrintInputShape(3)
        self.num_genres = num_genres

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        gnr = self.u2gnr[user]
        
        tokens = []
        labels = []
        genres = []
        for s, g in zip(seq, gnr):
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.seq_mask_token)
                    genres.append(self.gnr_mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                    genres.append(self.rng.randint(1, self.num_genres))
                else:
                    tokens.append(s)
                    genres.append(g)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)
                genres.append(g)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        genres = genres[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        genres = [0] * mask_len + genres

        self.printer.print(np.array(tokens), 'tokens')
        self.printer.print(np.array(labels), 'labels')
        self.printer.print(np.array(genres), 'genres')

        return torch.LongTensor(tokens), torch.LongTensor(genres), torch.LongTensor(labels)



class BertEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, seq_mask_token, negative_samples, u2gnr, gnr_mask_token):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.seq_mask_token = seq_mask_token
        self.negative_samples = negative_samples

        self.u2gnr = u2gnr
        self.gnr_mask_token = gnr_mask_token

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        gnr = self.u2gnr[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)
        seq = self.padding_and_trim(seq, 'seq')
        gnr = self.padding_and_trim(gnr, 'gnr')

        return torch.LongTensor(seq), torch.LongTensor(gnr), torch.LongTensor(candidates), torch.LongTensor(labels)

    def padding_and_trim(self, seq, seq_type):
        if seq_type == 'seq':
            seq = seq + [self.seq_mask_token]
        elif seq_type == 'gnr':
            seq = seq + [self.gnr_mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        return seq