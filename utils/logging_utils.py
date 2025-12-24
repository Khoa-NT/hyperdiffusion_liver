"""
This file contains some useful functions for logging.

Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Updated: 2024-11-30
"""

from pathlib import Path
import logging
import sys
import datetime


class get_logger:
    def __init__(self, name=None, input_logger=None, path=None):
        if input_logger is None:
            if name is None:
                name = str(datetime.datetime.now().strftime('%Y-%m-%d_%Hh-%Mm-%Ss'))

            ## ----- For logging to file ----- ##
            if path is None:
                path = Path.cwd()
            log_path = Path(path, name + ".log") # path/name.log
            if log_path.exists():
                log_path.unlink()
            file_handler = logging.FileHandler(filename=log_path)
            file_handler_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s]: %(message)s',
                                                       datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_handler_formatter)
            file_handler.setLevel(logging.DEBUG)

            ## ----- For logging to Terminal ----- ##
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler_formatter = logging.Formatter('[%(name)s][%(levelname)s]: %(message)s',
                                                         datefmt='%Y-%m-%d %H:%M:%S')
            stream_handler.setFormatter(stream_handler_formatter)
            stream_handler.setLevel(logging.DEBUG)

            ## ----- Add to logger ----- ##
            self.logger = logging.getLogger(name)
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

            ### Because Hydra is using root logging. So we have to stop the propagate here.
            self.logger.propagate = False

        else:
            # In this case, input_logger is given
            self.logger = input_logger

        
    def title(self, text, n_dash=30):
        self.logger.info(f"### {'-'*n_dash} {text} {'-'*n_dash} ###")

    def heading(self, text, n_dash=5):
        self.logger.info(f"## {'-'*n_dash} {text} {'-'*n_dash} ##")

    def msg_box(self, text):
        self.logger.info(f"#{'-'*60}#")

        for i_text in text.split(". "):
            self.logger.info(f"#\t{i_text}")

        self.logger.info(f"#{'-'*60}#")

    def info(self, text):
        self.logger.info(text)

    def info_dict(self, text, input_dict):
        self.logger.info(text)
        for k,v in input_dict.items():
            if isinstance(v, float):
                self.logger.info(f'\t{k}: {v:.6f}')
            else:
                self.logger.info(f'\t{k}: {v}')

    def debug(self, text):
        self.logger.debug(text)

    def error(self, text):
        self.logger.error(text)


