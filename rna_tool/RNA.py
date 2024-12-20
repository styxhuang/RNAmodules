import sys, os
sys.path.append(os.path.dirname(__file__))
import time
import logging
import os
from pathlib import Path
import shutil
import argparse

from utils.logger import rna_logger

class RNA_tool():
    def __init__(self, config_file: str, out_dir: str, \
                    is_train: bool=False, is_predict: bool=False, debug: bool=False):
        config_file = str(Path(config_file).resolve())
        out_dir = str(Path(out_dir).resolve())
        if debug:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        self.logger = rna_logger(out_dir, level=log_level)

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

def main():
    # 设置参数解析
    parser = argparse.ArgumentParser(description="Run RNAfold with specified parameters.")
    parser.add_argument("-i", "--input", type=str, help="RNA config yaml file", required=True, metavar=".fasta file")
    parser.add_argument("-o", "--output", type=str, help="Output dir", required=True, metavar="directory path")
    parser.add_argument("-t", "--train", help="If need to train", action="store_true")
    parser.add_argument("-p", "--predict", help="If need to predict", action="store_true")
    parser.add_argument("-v", "--verbose", help="Enable verbose mode", action="store_true")

    parsed_args = parser.parse_args()
    config_file     = parsed_args.input
    output_dir      = parsed_args.output
    train           = parsed_args.train
    predict         = parsed_args.predict
    verbose         = parsed_args.verbose

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    RNA_tool(config_file, output_dir, is_train=train, is_predict=predict, debug=verbose)

if __name__ == "__main__":
    main()