import pathlib
import sys
current_path = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(current_path))

from model import Model
from lib.memory_limiter import BatchMemoryLimiter

memory_limiter = BatchMemoryLimiter(Model)
memory_limiter.learn()
