import sys, os
sys.path.append(os.path.abspath(".."))

from lib.settings import BaseSettings

class ModelSettings(BaseSettings):
    MAX_BATCH_SIZE: 44