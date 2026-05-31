import os

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from gigashuffle import DataloaderConfig, MultiprocessShuffledDataloader


class TorchrunDataset(IterableDataset):
  def __iter__(self):
    worker_info = get_worker_info()
    worker_id = -1 if worker_info is None else worker_info.id
    num_workers = 1 if worker_info is None else worker_info.num_workers
    while True:
      yield [{'x': torch.arange(4) + worker_id, 'worker_id': torch.full((4,), worker_id), 'num_workers': torch.full((4,), num_workers)}]


def main() -> None:
  dist.init_process_group(backend='gloo')
  queue_name = os.environ['GIGASHUFFLE_QUEUE']
  loader = MultiprocessShuffledDataloader(
    TorchrunDataset(),
    DataloaderConfig(
      bs=4,
      shuffle_size=32,
      min_mixing=0.0,
      num_writers=1,
      num_readers=1,
      local_rank=int(os.environ['LOCAL_RANK']),
      global_rank=int(os.environ['RANK']),
      local_world_size=int(os.environ.get('LOCAL_WORLD_SIZE', '1')),
      global_world_size=int(os.environ['WORLD_SIZE']),
      redis_host=os.environ.get('REDIS_HOST', 'localhost'),
      redis_port=int(os.environ.get('REDIS_PORT', '6379')),
      redis_db=int(os.environ.get('REDIS_DB', '6')),
      queue_name=queue_name,
    ),
  )
  try:
    dummy = loader.get_dummy_batch()
    batch = next(iter(loader))
    assert dummy[0]['num_workers'].eq(2).all()
    assert batch[0]['x'].shape == (4,)
    dist.barrier()
  finally:
    if dist.is_initialized():
      dist.destroy_process_group()


if __name__ == '__main__':
  main()
