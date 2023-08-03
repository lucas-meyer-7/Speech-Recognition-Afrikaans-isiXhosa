import numpy as np

SR = 48000

DATA_PATH = 'data/high-quality-tts-data'
AF_PATH = 'af_za/za/afr'
XH_PATH = 'xh_za/za/xho'

class DataNotDownloadedError(Exception):
    def __init__(self, message="Data has not been downloaded yet."):
        self.message = message
        super().__init__(self.message)