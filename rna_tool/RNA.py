import sys, os
sys.path.append(os.path.dirname(__file__))
import time
import logging
import os
from pathlib import Path
import shutil
import argparse
import pickle

from utils.logger import rna_logger

class RNA_tool():
    def __init__(self, config_file: str, out_dir: str, \
                    is_train: bool=False, is_predict: bool=False, \
                    generate_feature: bool=False, debug: bool=False):
        config_file = str(Path(config_file).resolve())
        out_dir = str(Path(out_dir).resolve())
        if debug:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        self.logger = rna_logger(out_dir, level=log_level)
        
        # 只在指定-g参数时生成feature.info
        if generate_feature:
            self._generate_feature_info(config_file, out_dir)

        if is_train:
            self.logger.info(f"Start training model")
            from RNAtrain import train
            _train_dir = Path(out_dir).resolve() / 'train_results'
            _train_dir.mkdir()
            train(config_file, str(_train_dir), verbose=debug, logger=self.logger)
            self.logger.info(f'Finish training model. Results saved in {str(_train_dir)}')
        if is_predict:
            self.logger.info(f"Start predicting fasta")
            from RNApredict import predict
            _predict_dir = Path(out_dir).resolve() / 'predict_results'
            if is_train:
                _train_dir = _train_dir
            else:
                _train_dir = ''
            _predict_dir.mkdir()
            predict(config_file, str(_predict_dir), str(_train_dir), logger=self.logger)
            self.logger.info(f'Finish predicting. Results saved in {str(_predict_dir)}')
        print()
    def _init_logger(self, **kwargs):
        log_type = kwargs.get("log_type", "simple")
        log_level = kwargs.get("log_level", "debug")
        out_dir     = kwargs.get("output")
        if log_type == "simple":
            FORMAT = "%(message)s"
        else:
            FORMAT = "%(asctime)s %(levelname)-2s %(name)-10s %(message)s"

        TIME_FMT = "%m/%d %H:%M:%S"
        fmt = logging.Formatter(fmt=FORMAT, datefmt=TIME_FMT)
        log_name = f'{out_dir}/runtime.log'

        if log_level == "debug":
            level = logging.DEBUG
        elif log_level == "info":
            level = logging.INFO
        elif log_level == "warning":
            level = logging.WARNING
        elif log_level == 'error':
            level = logging.ERROR
        else:
            level = logging.INFO
        _logger = logging.getLogger('runtime')
        _logger.setLevel(level)

        # logging file handler
        fh = logging.FileHandler(log_name, mode='w')
        fh.setLevel(level)
        fh.setFormatter(fmt)
        _logger.addHandler(fh)
        return _logger

    def _generate_feature_info(self, config_file: str, out_dir: str):
        """生成feature.info文件"""
        import yaml
        import pandas as pd
        
        try:
            # 读取配置文件
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # 获取feature.yaml的路径
            feature_yaml = config.get('feature_yaml', '')
            if not feature_yaml:
                self.logger.error("No feature.yaml path specified in config file")
                return
            
            feature_yaml_path = Path(feature_yaml)
            if not feature_yaml_path.exists():
                self.logger.error(f"Feature yaml file not found: {feature_yaml_path}")
                return
            
            # 读取feature.yaml
            with open(feature_yaml_path, 'r') as f:
                feature_config = yaml.safe_load(f)
            
            features = feature_config.get('features', {})
            
            # 构造feature_list从features的type获取
            feature_list = []
            params = []
            extract_feature = {}
            
            # 按顺序处理每个feature
            for feature_name, feature_info in features.items():
                # 收集feature类型
                feature_type = feature_info['type']
                feature_list.append(feature_type)
                
                # 收集参数
                feature_params = feature_info['params']
                # 获取第一个参数值（假设每个feature只需要一个主要参数）
                param_value = next(iter(feature_params.values()))
                params.append(param_value)
                
                # 构造feature字典
                feature_dict = {
                    'type': feature_type,
                    'params': feature_params,
                    'data': None  # 初始化data字段
                }
                
                # 读取CSV文件
                try:
                    csv_path = feature_info['csv']
                    df = pd.read_csv(csv_path)
                    feature_dict['data'] = df  # 将DataFrame存储在data字段
                    extract_feature[feature_name] = feature_dict
                except Exception as e:
                    self.logger.warning(f"无法读取特征文件 {csv_path}: {str(e)}")
            
            # 构造完整的feature_info字典
            feature_info = {
                'feature_list': feature_list,
                'deep_fusion': feature_config.get('fusion', [])[0].split(', '),  # 将字符串分割成列表
                'params': params,
                'random_seed': feature_config.get('random_seed', []),
                'dataset': config.get('train_fasta', ''),  # 从主配置文件获取数据集路径
                'extract_feature': extract_feature  # 现在每个feature包含type、params和data
            }
            
            # 使用pickle保存feature.info文件
            feature_info_path = Path(out_dir) / 'feature.info'
            with open(feature_info_path, 'wb') as f:
                pickle.dump(feature_info, f)
            
            self.logger.info(f"Feature info file generated at {feature_info_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating feature info: {str(e)}")
            raise

def main(args=None):
    import warnings
    warnings.filterwarnings("ignore", message="Numerical issues were encountered when scaling the data")

    # 设置参数解析
    parser = argparse.ArgumentParser(description="Run RNAfold with specified parameters.")
    parser.add_argument("-i", "--input", type=str, help="RNA config yaml file", required=True, metavar=".fasta file")
    parser.add_argument("-o", "--output", type=str, help="Output dir", required=True, metavar="directory path")
    parser.add_argument("-t", "--train", help="If need to train", action="store_true")
    parser.add_argument("-p", "--predict", help="If need to predict", action="store_true")
    parser.add_argument("-g", "--generate", help="Generate feature.info file", action="store_true")
    parser.add_argument("-v", "--verbose", help="Enable verbose mode", action="store_true")

    # 如果没有提供参数，则使用sys.argv[1:]
    parsed_args = parser.parse_args(args)
    config_file     = parsed_args.input
    output_dir      = parsed_args.output
    train           = parsed_args.train
    predict         = parsed_args.predict
    generate        = parsed_args.generate
    verbose         = parsed_args.verbose

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    RNA_tool(config_file, output_dir, is_train=train, is_predict=predict, 
             generate_feature=generate, debug=verbose)
    return parsed_args

if __name__ == "__main__":
    main()  # 直接运行时不传参数，使用命令行参数