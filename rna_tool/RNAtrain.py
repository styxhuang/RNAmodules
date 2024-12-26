import sys, os
import argparse
sys.path.append(os.path.dirname(__file__))
from pathlib import Path
import logging
import random
from itertools import chain
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten,Bidirectional
from keras.layers import LSTM
from attention import Attention

from utils.config_loader import load_config
from utils.timer import Timer, export_time
from utils.logger import rna_logger

from src.Kmer import Kmer
from src.PCP import pcp
from src.PseDNC import psednc
from src.att_lstm_model import att_model, lstm_model
from src.RNAfold import RNAfold
from src.KNFC import generate_fn_file
from src.train import rna_train

class RNAtrain:
    def __init__(self, config: dict, output_dir: str, mode: bool, logger: logging):
        self.logger = logger

        _train_fasta    = Path(config['data']['train_fasta']).resolve()
        _pcp_file       = Path(config['data']['pcp_file']).resolve()
        self.train_fasta = self._read_fasta(str(_train_fasta.resolve()))
        self.pcp = str(_pcp_file)
        self.fusion = config['fusion']

        # chdir
        _pre_dir = os.getcwd()
        os.chdir(output_dir)

        # features
        self.features = {}
        with Timer('Train.extract_feature') as t1:
            self._extract_feature(config)

        # fusion
        self.feature_list   = []
        self.params         = []
        self.deep_fusion    = []
        self.random_seed    = []
        self.summary_df     = []
        self.logger.info('Extracted feature!')
        with Timer('Train.fusion_data') as t2:
            self._process_fusion()

        self.logger.info('Fusioned data!')
        # training
        with Timer('Train.train') as t3:
            df = rna_train(self.feature_list, self.deep_fusion, self.params, self.random_seed,
                    self.train_fasta, self.summary_df, config, self.logger)
        self.logger.info('Trained Data!')
        os.chdir(_pre_dir)

    def _read_fasta(self, file):
        f = open(file)
        documents = f.readlines()
        string = ""
        flag = 0
        fea = []
        for document in documents:
            if document.startswith(">") and flag == 0:
                # if document.endswith(">") and flag == 0:
                flag = 1
                continue
            elif document.startswith(">") and flag == 1:
                # elif document.endswith(">") and flag == 1:
                string = string.upper()
                fea.append(string)
                string = ""
            else:
                string += document
                string = string.strip()
                string = string.replace(" ", "")

        fea.append(string)
        f.close()
        return fea

    def _extract_feature(self, config: dict):
        if len(config.keys()) == 0:
            raise ValueError('Fail to find any keys!')
        else:
            for f, v in config['features'].items():
                self.logger.info(f'Extract feature: {v["type"]}')
                self.features[f] = v
                # with Timer(f'Train.extract_feature.{f}_{v["type"]}') as _t:
                if v['type'] == 'kmer':
                    _data = Kmer(self.train_fasta, v['params']['k'])
                elif v['type'] == 'knfc':
                    _data = generate_fn_file(self.train_fasta, v['params']['k'])
                elif v['type'] == 'pcp':
                    _data = pcp(self.train_fasta, v['params']['k'], self.pcp)
                elif v['type'] == 'pseDNC':
                    _data = psednc(self.train_fasta, v['params']['k'], self.pcp)
                elif v['type'] == 'rnafold':
                    _data = RNAfold(self.train_fasta)
                else:
                    ...
                self.features[f]['data'] = _data

    def _process_fusion(self):
        def _att_model(seed: int, df: pd.DataFrame):
            random.seed(seed)
            tf.random.set_seed(seed)
            model = Sequential()
            model.add(Attention())
            model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
            data = np.array(df)
            data = np.nan_to_num(data, nan=-9999)
            data = data[:, 0:]
            data = data.astype(np.float32)
            [m1, n1] = np.shape(data)
            X = np.reshape(data, (-1, 1, n1))
            cv_clf = model
            tf.config.experimental_run_functions_eagerly(False)
            feature = cv_clf.predict(X)
            data_csv = pd.DataFrame(data=feature)
            return data_csv

        def _lstm(seed: int, df: pd.DataFrame):
            # add random seed for model
            random.seed(seed)
            tf.random.set_seed(seed)
            model = Sequential()
            model.add(Bidirectional(LSTM(200)))
            model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
            data = np.array(df)
            data = np.nan_to_num(data, nan=-9999)
            data = data[:, 0:]
            data = data.astype(np.float32)
            [m1, n1] = np.shape(data)
            # shu = scale(data)
            # X1 = shu

            X = np.reshape(data, (-1, 1, n1))
            cv_clf = model
            tf.config.experimental_run_functions_eagerly(False)
            feature = cv_clf.predict(X)

            data_csv = pd.DataFrame(data=feature)
            return data_csv

        pre_fusion_data = []
        for _lyr, v in self.fusion.items(): # fusion by layer
            _pre_fusion_data = []
            fusion_type = v['type']
            feature_names = v['layers']
            seed = v['seed']
            for _fea in feature_names:
                if fusion_type == 'attention':
                    _df = _att_model(seed, self.features[_fea]['data'])
                elif fusion_type == 'lstm':
                    _df = _lstm(seed, self.features[_fea]['data'])
                else:
                    raise RuntimeError(f'Unexpected fusion type <{fusion_type}>!')
                self.feature_list.append(self.features[_fea]['type'])
                self.params.append(self.features[_fea]['params'].get('k', 1))
                self.deep_fusion.append(fusion_type)
                self.random_seed.append(seed)
                _pre_fusion_data.append(_df)
            _ddf = pd.concat(_pre_fusion_data, axis=1)
            pre_fusion_data.append(_ddf)

        # fusion all features
        self.summary_df = pd.concat(pre_fusion_data, axis=1)

def train(config_file: str, output_dir: str, verbose=None, logger=None):
    config = load_config(config_file)
    if not logger:
        raise ValueError("No logger found!!")

    with Timer('Train') as timer:
        rna = RNAtrain(config, output_dir, verbose, logger)
        
        # save feature.info
        _out_data = {
            'feature_list': rna.feature_list,
            'deep_fusion': rna.deep_fusion,
            'params': rna.params,
            'random_seed': rna.random_seed,
            'dataset': config['data']['train_fasta'],
            'extract_feature': pd.DataFrame(rna.features)
        }
        _feature_info_path = os.path.join(output_dir, 'feature.info')
        with open(_feature_info_path, 'wb') as file:
            pickle.dump(_out_data, file)

    export_time(logger, log_type='a')
    return config
