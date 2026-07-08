from dataclasses import dataclass


@dataclass
class DataloaderConfig:
  bs: int
  shuffle_size: int
  min_mixing: float = 0.5
  num_writers: int = 2
  num_readers: int = 2
  writer_max_retries: int = 100
  fill_once: bool = False
  evict_on_read: bool = True
  local_rank: int = 0
  global_rank: int = 0
  local_world_size: int = 1
  global_world_size: int = 1
  queue_name: str = ''
