import time
from collections import defaultdict
import sys
import os
import logging
from pathlib import Path

import networkx as nx

class Ticks(object):
    _instance = None
    ticks = defaultdict(lambda: 0)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance


class Timer:
    def __init__(self, name: str) -> None:
        self.start_time = 0
        self.end_time = 0
        self.execution_time = 0
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        Ticks().ticks[self.name] = self.execution_time


class time_logger:
    def __init__(self) -> None:
        FORMAT = "%(asctime)s -%(levelname)-2s- %(message)s"
        TIME_FMT = "%m/%d %H:%M:%S"
        log_path = Path(os.getcwd())
        if not log_path.exists():
            log_path.mkdir(parents=True)
        log_path = str(log_path)

        name = 'time_record'
        fmt = logging.Formatter(fmt=FORMAT, datefmt=TIME_FMT)
        log_name = os.path.join(log_path, f'{name}.log')
        self.log_name = log_name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # logging file handler
        file_handler = logging.FileHandler(log_name, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)


def _format_time(elapsed_time: float) -> str:
    if elapsed_time < 60:
        formatted_time = f"{elapsed_time:.2f} sec"
    else:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        formatted_time = f"{minutes} min and {seconds:.2f} sec"
    return formatted_time


def export_time(log=None, log_file: str = None, log_type='w'):
    time_tree = nx.DiGraph()
    time_tree.add_node('')
    time_tree.nodes['']['cum'] = 0

    if log is None:
        logger = time_logger().logger
    else:
        logger = log

    for k, v in Ticks.ticks.items():
        time_tree.nodes['']['cum'] += v

        nodes = k.strip('.').split('.')
        previous_node = ''

        for node in nodes:
            if node not in time_tree.nodes:
                time_tree.add_node(node)
                time_tree.add_edge(previous_node, node)

            previous_node = node
            time_tree.nodes[node]['cum'] = time_tree.nodes[node].get(
                'cum', 0) + v

    name_col = []
    time_col = []

    def iterate_tree(node, indent):
        nonlocal time_tree, name_col, time_col

        prefix = ['    '] * indent
        name_col.append(''.join(prefix + [node]))
        time_col.append(time_tree.nodes[node]['cum'])

        for successor in time_tree.successors(node):
            iterate_tree(successor, indent + 1)

    iterate_tree('', 0)
    name_col[0] = 'TOTAL'
    max_name = max([len(i) for i in name_col])
    width = max_name + 4 + 8 + 1 + 2 + 4 + 6

    log_msg = '\n'

    log_msg += '=' * width + '\n'
    log_msg += '   < MaxFlow RNA Component Timing Summary >' + '\n'
    log_msg += '=' * width + '\n'
    for name, timing in zip(name_col, time_col):
        format_time = _format_time(timing)
        log_msg += name.ljust(max_name + 4)
        log_msg += f'{format_time}' + '\n'

    log_msg += '=' * width + '\n'
    log_msg += '   < MaxFlow RNA Component Timing Summary >' + '\n'
    log_msg += '=' * width + '\n'
    logger.info(log_msg)

    if log_file is not None:
        with open(log_file, log_type) as f:
            f.write(log_msg)
