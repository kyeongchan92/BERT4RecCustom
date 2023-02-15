from pathlib import Path
import os
import tempfile
import shutil
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from datetime import date

from .utils import *
from config import RAW_DATASET_ROOT_FOLDER


class ML1MDataset:
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split
        if args.use_attributes:
            self.use_attributes = args.use_attributes
            self.attributes = args.attributes

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    def raw_code(cls):
        print(f"raw_code's return : {cls.code()}")
        return cls.code()

    @classmethod
    def code(cls):
        return 'ml-1m'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'movies.dat',
                'ratings.dat',
                'users.dat']

    @classmethod
    def is_zipfile(cls):
        return True

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, sep='::', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df

    def load_movies_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('movies.dat')
        df = pd.read_csv(file_path, sep='::', header=None, encoding='ISO-8859-1')
        df.columns = ['sid', 'title', 'genre']
        return df

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def load_attrs_df(self, rating_df):
        '''
        Return : 
            attr_df = itemid |   genre  |  ...
                        1    | 'comedy' |
        '''
        attr_df = self.load_movies_df()
        return attr_df[['sid'] + self.attributes]

    # def get_i2attr_map(self, attrs_df):


    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        df = self.make_implicit(df)
        df = self.filter_triplets(df)
        df, u_i_map = self.densify_index(df, target_cols=['sid', 'uid'])
        tra_item_seq, val_item_seq, tes_item_seq = self.split_tra_val_tes(df, len(u_i_map['uid']))

        # attributes
        i2attr_map = []
        if self.use_attributes:
            attrs_df = self.load_attrs_df(df)
            attrs_df['sid'] = attrs_df['sid'].map(u_i_map['sid'])
            attrs_df, amap = self.densify_index(attrs_df, target_cols=self.attributes)
            for attr_name in self.attributes:
                i2attr_map.append(dict(zip(attrs_df['sid'], attrs_df[attr_name])))
        else:
            amap = {}

        '''
        amap : nested dict
        ex)
        {
            'attr1' : dict,
            'attr2' : dict
            }
        - amap is also nested dict
        when not using attributes, amap = {}
        '''

        dataset = {'tra_item_seq': tra_item_seq,  # dict : {densified user's id : [3, 4, ...](densified item's id)}
                   'val_item_seq': val_item_seq,
                   'tes_item_seq': tes_item_seq,
                   'u_i_map' : u_i_map,
                   'i2attr_map' : i2attr_map,
                   'amap' : amap 
                   }
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        print("Raw file doesn't exist. Downloading...")
        if self.is_zipfile():
            tmproot = Path(tempfile.mkdtemp())
            tmpzip = tmproot.joinpath('file.zip')
            tmpfolder = tmproot.joinpath('folder')
            download(self.url(), tmpzip)
            unzip(tmpzip, tmpfolder)
            if self.zip_file_content_is_folder():
                tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
            shutil.move(tmpfolder, folder_path)
            shutil.rmtree(tmproot)
            print()
        else:
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(self.url(), tmpfile)
            folder_path.mkdir(parents=True)
            shutil.move(tmpfile, folder_path.joinpath('ratings.csv'))
            shutil.rmtree(tmproot)
            print()

    def make_implicit(self, df):
        print('Turning into implicit ratings')
        df = df[df['rating'] >= self.min_rating]
        # return df[['uid', 'sid', 'timestamp']]
        return df

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]

        return df

    def densify_index(self, df, target_cols=['sid', 'uid']):
        print('Densifying index')
        maps = {}
        for col in target_cols:
            _map = {u: i for i, u in enumerate(set(df[col]), start=1)}
            maps[col] = (_map)
            # smap = {s: i for i, s in enumerate(set(df['sid']))}
            df[col] = df[col].map(_map)
            
        return df, maps


    def split_tra_val_tes(self, df, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(lambda u_df: list(u_df.sort_values(by='timestamp')['sid']))
            # user2genres = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['gid']))

            train, val, test = {}, {}, {}

            # train_g, val_g, test_g = {}, {}, {}
            for user in range(1, user_count+1):
                items = user2items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]

                # genres = user2genres[user]
                # train_g[user], val_g[user], test_g[user] = genres[:-2], genres[-2:-1], genres[-1:]

            return train, val, test
        # elif self.args.split == 'holdout':
        #     print('Splitting')
        #     np.random.seed(self.args.dataset_split_seed)
        #     eval_set_size = self.args.eval_set_size

        #     # Generate user indices
        #     permuted_index = np.random.permutation(user_count)
        #     train_user_index = permuted_index[                :-2*eval_set_size]
        #     val_user_index   = permuted_index[-2*eval_set_size:  -eval_set_size]
        #     test_user_index  = permuted_index[  -eval_set_size:                ]

        #     # Split DataFrames
        #     train_df = df.loc[df['uid'].isin(train_user_index)]
        #     val_df   = df.loc[df['uid'].isin(val_user_index)]
        #     test_df  = df.loc[df['uid'].isin(test_user_index)]

        #     # DataFrame to dict => {uid : list of sid's}
        #     train = dict(train_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
        #     val   = dict(val_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
        #     test  = dict(test_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
        #     return train, val, test
        else:
            raise NotImplementedError

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = 'min_rating{}-min_uc{}-min_sc{}-split{}' \
            .format(self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')


