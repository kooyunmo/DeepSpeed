from deepspeed.utils.logging import logger, log_dist
from deepspeed.runtime.dataloader import RepeatingLoader

from .measure import event_manager