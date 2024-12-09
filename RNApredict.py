import sys, os
sys.path.append(os.path.dirname(__file__))
import time
import logging
import pickle
import os
from pathlib import Path

import argparse
import numpy as np
import pandas as pd
from keras.models import Model, load_model
from sklearn.preprocessing import scale

from src.Kmer import Kmer
from src.PCP import pcp
from src.PseDNC import psednc
from src.att_lstm_model import att_model, lstm_model
from src.RNAfold import RNAfold
from src.KNFC import generate_fn_file
from utils.config_loader import load_config

class RNAPredict():
    def __init__(self, fasta_file: str, output_dir: str, train_file: str, feature_file: str, model_file: str, pcp_file: str, mode=False, logger=None) -> None:
        self.fasta_file     = fasta_file
        self.output_dir     = output_dir
        self.train_file     = train_file
        self.feature_info   = pickle.load(open(feature_file, 'rb'))
        self.model_file     = model_file
        self.logger         = logger
        self.debug          = mode

        # basic properties
        self.feature_list = self.feature_info.get("feature_list")
        self.feature_data = self.feature_info.get("extract_feature")
        self.pcp = pcp_file
        self.dataset_feature_extraction = []

        # fasta
        self.train_fasta    = self._read_dataset(self.train_file) # 训练用fasta
        self.predict_fasta  = self._read_fasta(fasta_file) # 预测用fasta
        self.model          = load_model(self.model_file)

        self.data_list = []
        self.out_index_list =[]
        self.out_init_list =[]
        self.out_prob_list =[]
        self.out_modify_list =[]
        self.out_name_list =[]
        self.out_sequence_list =[]

        # 正式运行
        self.predict()

    # read dataset
    def _read_dataset(self, file):
        f = open(file)
        documents = f.readlines()
        string = ""
        flag = 0
        fea = []
        for document in documents:
            if document.startswith(">") and flag == 0:
                flag = 1
                continue
            elif document.startswith(">") and flag == 1:
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

    # 读取原始序列切段41bp
    def _read_fasta(self,file):
        fasta = {}

        with open(file) as file_one:
            for line in file_one:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    active_sequence_name = line[1:]
                    if active_sequence_name not in fasta:
                        fasta[active_sequence_name] = []
                    continue
                sequence = line
                fasta[active_sequence_name].append(sequence)
                
        return fasta

    def _extract_feature(self):
        self.logger.info(f'Obtain train dataset feature extraction')
        for _, _f in enumerate(self.feature_list):
            for _, v in self.feature_data.items():
                if v['type'] == _f:
                    self.dataset_feature_extraction.append(v['data'])

    def _rna_predict(self):
        def cutArray(sequence):
            cut = []
            for i in range(len(sequence) - 41 + 1):
                cut.append(sequence[i:i + 41])
            return cut

        t0 = time.time()
        for n, name in enumerate(self.predict_fasta):
            self.logger.info(f'START PREDICT SEQ: <{name}>')
            self.logger.info(f'\t => Iter {n} start:')
            self.logger.info(f'\t\t => Start Extract feature:')

            t_predict_fasta = time.time()
            sequence = self.predict_fasta[name] = "".join(self.predict_fasta[name])
            cut0 = cutArray(sequence)
            cut = self.train_fasta + cut0
            fusion = pd.DataFrame()

            for i in range(len(self.feature_list)):
                cut0path = pd.DataFrame()
                featurepath = pd.DataFrame()
                fusionpath = pd.DataFrame()

                # 计算测试数据特征提取数据
                if self.feature_list[i] == 'kmer':
                    k0 = time.time()
                    k = int(self.feature_info['params'][i])
                    cut0path = Kmer(cut0, k)
                    k1 = time.time()
                elif self.feature_list[i] == 'knfc':
                    k0 = time.time()
                    k = int(self.feature_info['params'][i])
                    cut0path = generate_fn_file(cut0, k)
                    k1 = time.time()
                elif self.feature_list[i] == 'psednc':
                    k0 = time.time()
                    k = int(self.feature_info['params'][i])
                    cut0path = psednc(cut0, k, self.pcp)
                    k1 = time.time()
                elif self.feature_list[i] == 'pcp':
                    k0 = time.time()
                    k = int(self.feature_info['params'][i])
                    cut0path = pcp(cut0, k, self.pcp)
                    k1 = time.time()
                elif self.feature_list[i] == 'RNAfold':
                    k0 = time.time()
                    cut0path = RNAfold(cut0)
                    k1 = time.time()
                self.logger.info(f'\t\t\t => feature <{self.feature_list[i]}> costs: {round(k1 - k0, 2)}s')
                if self.debug:
                    cut0path.to_csv(f'{name}_feature_{self.feature_list[i]}.csv')
                
                # 拼接训练集数据与预测集数据的特征提取
                featurepath = pd.concat([self.dataset_feature_extraction[i], cut0path])
                
                # 计算合并数据的深度融合结果
                random_seed = int(self.feature_info['random_seed'][i])
                fusion_type = self.feature_info["deep_fusion"][i]
                self.logger.info(f"\t\t\t\t => 第{i}次深度融合：{i}，深度融合为{fusion_type}，随机种子为{random_seed}")

                ls_at_t0 = time.time()
                if self.feature_info["deep_fusion"][i] == "lstm":
                    fusionpath = lstm_model(featurepath, random_seed)
                elif self.feature_info["deep_fusion"][i] == "attention":
                    fusionpath = att_model(featurepath, random_seed)
                else:
                    print("Error: No deep fusion after feature extraction knfc")
                ls_at_t1 = time.time()
                self.logger.info(f'\t\t\t => lstm_attention fusion costs: {round(ls_at_t1 - ls_at_t0, 2)}s')

                # 合并特征提取
                fusion = pd.concat([fusion, fusionpath], axis=1)
            t_extract_feature = time.time()
            predictdf = fusion
            shape = predictdf.shape
            self.logger.info(f"\t\t => The shape after deep fusion concat is {shape}")

            # 测试开始
            data=np.array(predictdf)
            data=np.nan_to_num(data, nan=-9999)
            data=data[:,0:]
            [m1,n1]=np.shape(data)
            # scale depends on y-axis which will be affected by training data above the predict data, only input predict
            #  data will cause predict error
            X = scale(data)
            X = X.reshape(m1, n1,1)
            y_score= self.model.predict(X, batch_size=2000)
            y_list = y_score[:,1].tolist()

            t_predict = time.time()
            # self.logger.info(f"索引: {name}, 突变概率: {y_list[len(self.train_fasta):][i]}")

            sub_level = []
            for i in range(len(y_list[len(self.train_fasta):])):
                seqindex = 21+i
                if cut[len(self.train_fasta):][i][20] == "C":
                    if y_list[len(self.train_fasta):][i]>=0.6:
                       output = {"索引":seqindex,"原始碱基":"C",'突变概率': y_list[len(self.train_fasta):][i],"突变碱基":"U"}
                       self.out_index_list.append(seqindex)
                       self.out_init_list.append("C")
                       self.out_prob_list.append(y_list[len(self.train_fasta):][i])
                       self.out_modify_list.append("U")
                       self.out_name_list.append(name)
                       self.out_sequence_list.append(sequence)
                    else:
                       output = {"索引":seqindex,"原始碱基": "C", '突变概率': y_list[len(self.train_fasta):][i], "突变碱基": "C"}
                       self.out_index_list.append(seqindex)
                       self.out_init_list.append("C")
                       self.out_prob_list.append(y_list[len(self.train_fasta):][i])
                       self.out_modify_list.append("C")
                       self.out_name_list.append(name)
                       self.out_sequence_list.append(sequence)
                else:
                    output = {"索引":seqindex,"原始碱基": cut[len(self.train_fasta):][i][20], '突变概率': "", "突变碱基": cut[len(self.train_fasta):][i][20]}
                    self.out_index_list.append(seqindex)
                    self.out_init_list.append(cut[len(self.train_fasta):][i][20])
                    self.out_prob_list.append("")
                    self.out_modify_list.append(cut[len(self.train_fasta):][i][20])
                    self.out_name_list.append(name)
                    self.out_sequence_list.append(sequence)
                sub_level.append(output)
            data_temp = {
                "structure_name": name,
                "structure_path": "",
                "sequence":sequence,
                "status": "Success",
                "error": "",
                "isResult": "True",
                'isStruct': 'False',
                "sub_level": sub_level
            }
            self.data_list.append(data_temp)
            t_out = time.time()
            self.logger.info(f'\t => Iter {n} cost: {round(t_out - t_predict_fasta, 2)}s')
            self.logger.info(f'\t\t => extract feature cost: {round(t_extract_feature - t_predict_fasta, 2)}s')
            self.logger.info(f'\t\t => predict cost: {round(t_predict - t_extract_feature, 2)}s')
            self.logger.info(f'\t\t => generate output cost: {round(t_out - t_predict, 2)}s')
            self.logger.info(f'\t {"="*34}')
        t1 = time.time()
        self.logger.info(f'Predict cost {round(t1 - t0, 2)}s')
        featureDf = pd.DataFrame({
            "特性提取": self.feature_list,
            "特征步长": self.feature_info["params"],
            "深度融合": self.feature_info["deep_fusion"],
            "融合随机种子": self.feature_info["random_seed"]
        }, columns=["特性提取", "特征步长", "深度融合", "融合随机种子"])

        dataDf_csv = pd.DataFrame({
            "structure_name":self.out_name_list,
            "sequence":self.out_sequence_list,
            "索引":self.out_index_list,
            "原始碱基":self.out_init_list,
            "突变概率":self.out_prob_list,
            "突变碱基":self.out_modify_list
        },columns=["structure_name","sequence","索引","原始碱基", "突变概率","突变碱基"])

        csv_feature_path = "result_feature.csv"
        csv_path = "result.csv"
        featureDf.to_csv(csv_feature_path, mode='w', index=False, encoding='utf_8_sig')
        dataDf_csv.to_csv(csv_path, mode='a', index=False, header=True, encoding='utf_8_sig')

    def predict(self):
        '''
        1. 初始化日志
        2. 提取的训练模型
        3. 预测fasta
        '''
        t1 = time.time()
        os.chdir(self.output_dir)
        self._extract_feature()
        t2 = time.time()
        self.logger.info(f'EXTRACT FEATURE TIME COST: {round(t2 - t1, 2)}s')
        self._rna_predict()
        t3 = time.time()
        self.logger.info(f'PREDICT TIME COST: {round(t3 - t2, 2)}s')

def predict(config_file: str, output_dir: str, train_dir: str='', verbose: bool=False, logger: logging=None):
    _config = load_config(config_file)
    _predict_fasta  = _config['data']['predict_fasta']
    _train_fasta    = _config['data']['train_fasta']
    _pcp_file       = _config['data']['pcp_file']
    if train_dir == '':
        _train_dir  = Path(_config['data']['train_dir'])
    else:
        _train_dir  = Path(train_dir)
    _feature_file = _train_dir / 'feature.info'
    _model_file = _train_dir / 'model.h5'
    # check above dir and files are exists
    # TODO: ...
    if not _feature_file.exists():
        raise FileNotFoundError(str(_feature_file))
    if not _model_file.exists():
        raise FileNotFoundError(str(_model_file))

    _rna = RNAPredict(_predict_fasta, output_dir, _train_fasta, _feature_file, _model_file, _pcp_file, mode=verbose, logger=logger)
    
def main(args=None):
    # 设置参数解析
    parser = argparse.ArgumentParser(description="Run RNAfold with specified parameters.")
    parser.add_argument("-i", "--input", type=str, help="Predict FASTA file", required=True, metavar=".fasta file")
    parser.add_argument("-o", "--output", type=str, help="Output dir", required=True, metavar="directory path")
    parser.add_argument("-t", "--train", type=str, help="Trained FASTA file", required=True, metavar=".fasta file")
    parser.add_argument("-f", "--feature", type=str, help="Input feature file, recording data used when trained model", required=True, metavar="xxx_feature.info file")
    parser.add_argument("-m", "--model", type=str, help="Input model", required=True, metavar=".h5 file")
    parser.add_argument("-p", "--pcp", type=str, help="physical_chemical_properties_RNA", required=True, metavar="")
    parser.add_argument("-v", "--verbose", help="Enable verbose mode", action="store_true")

    parsed_args = parser.parse_args(args)
    fasta_file      = parsed_args.input
    output_dir      = parsed_args.output
    train_file      = parsed_args.train
    feature_file    = parsed_args.feature
    model_file      = parsed_args.model
    pcp_file        = parsed_args.pcp
    verbose         = parsed_args.verbose

    # 执行命令
    RNAPredict(fasta_file, output_dir, train_file, feature_file, model_file, pcp_file, mode=verbose)

    return parsed_args

if __name__ == '__main__':
    main()
