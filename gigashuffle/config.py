from dataclasses import dataclass


@dataclass
class DataloaderConfig:
  bs: int
  shuffle_size: int
  min_mixing: float = 0.5
  num_writers: int = 2
  num_readers: int = 2
  local_rank: int = 0
  global_rank: int = 0
  local_world_size: int = 1
  global_world_size: int = 1
  redis_host: str = 'localhost'
  redis_port: int = 6379
  redis_db: int = 6
  shm_dir: str = '/dev/shm'
  queue_name: str = ''
