import os
from pathlib import Path
import logging
import inspect

class rna_logger:
    _instance = None

    def __new__(cls, log_path: str, level=logging.DEBUG) -> None:
        if cls._instance is None:
            cls._instance = super(rna_logger, cls).__new__(cls)
            cls._instance._initialize(log_path, level)
        return cls._instance

    def _initialize(self, log_path: str, level=logging.DEBUG) -> None:
        if level == logging.DEBUG:
            FORMAT = "%(asctime)s -%(levelname)-2s- %(message)s"
            TIME_FMT = "%m/%d %H:%M:%S"
        else:
            FORMAT = "%(message)s"
            TIME_FMT = "%m"
        log_path = Path(log_path)
        if not log_path.exists():
            try:
                log_path.mkdir(parents=True)
            except PermissionError:
                log_path = os.getcwd()
        log_path = str(log_path)
        fmt = logging.Formatter(fmt=FORMAT, datefmt=TIME_FMT)
        log_name = os.path.join(log_path, 'runtime.log')
        self.log_name = log_name
        self.logger = logging.getLogger('runtime')
        self.logger.setLevel(level)
        self.logger.log_path = log_path

        # logging file handler
        file_handler = logging.FileHandler(log_name)
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)

        with open(log_name, 'w') as f:
            header_msg = self.__logHeader('Start RNA modules', log_path)
            f.write(header_msg)

    def __logHeader(self, header, logPath):
        box_width = 120
        header_msg = f'LOGGING {header} MODULE'

        # Calculate the space at both sides
        padding = ((box_width - len(header_msg)) // 2) - 2

        # Generate the boxed msg
        centered_message = f"{' ' * padding} {header_msg} {' ' * padding}"
        _DIR_msg = f"{' ' * (padding-4)} WORKING DIR:      {'/'.join(Path(logPath).parts[-2:])}"
        _msg = [
            '=' * box_width, centered_message, _DIR_msg,
            '=' * box_width
        ]
        box = '\n'.join(_msg) + '\n'
        return box

    def __procMsg(self, frame, msg):
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        filename = '/'.join(Path(filename).parts[-2:])
        _msg = f'{filename}:{lineno}: {msg}'
        return _msg

    def debug(self, msg):
        # Get the frame information for the caller of the current function
        frame = inspect.currentframe().f_back
        _msg = self.__procMsg(frame, msg)
        self.logger.debug(_msg)

    def info(self, msg):
        frame = inspect.currentframe().f_back
        _msg = self.__procMsg(frame, msg)
        self.logger.info(_msg)

    def warning(self, msg):
        frame = inspect.currentframe().f_back
        _msg = self.__procMsg(frame, msg)
        self.logger.warning(_msg)

    def error(self, msg):
        frame = inspect.currentframe().f_back
        _msg = self.__procMsg(frame, msg)
        self.logger.error(_msg)

    def fatal(self, msg):
        frame = inspect.currentframe().f_back
        _msg = self.__procMsg(frame, msg)
        self.logger.fatal(_msg)

    def critical(self, msg):
        frame = inspect.currentframe().f_back
        _msg = self.__procMsg(frame, msg)
        self.logger.critical(_msg)
