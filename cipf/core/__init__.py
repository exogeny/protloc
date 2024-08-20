import dataclasses
from typing import Mapping, Optional

from cipf.core.config_parser import parse_configuration
from cipf.core.config_parser import serialize_configuration
from cipf.core.intializer import init_scheduler_from_config
from cipf.core.intializer import init_optax_from_config
from cipf.core.intializer import init_model_from_config
from cipf.core.intializer import init_initial_state
from cipf.core.intializer import get_cls_from_target


@dataclasses.dataclass
class OptimizationConfig:
  constructor: str = 'optax.adam'
  parameters: str = ''


@dataclasses.dataclass
class LearningRateConfig:
  num_epochs: int
  base_learning_rate: float
  learning_schedule: str = 'piecewise'
  warmup_epochs: Optional[int] = 0
  boundaries_and_scales: Optional[Mapping[int, float]] = None


@dataclasses.dataclass
class AugmentConfig:
  """Dataset augmentation configuration."""
  name: Optional[str] = ''
  cutout_const: int = 0
  translate_const: int = 250
