import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
  rank = int(os.environ['RANK'])
  world_size = int(os.environ['WORLD_SIZE'])
  # only rank 0 spawns a writer; rank 1 spawns none. total_writers=1 across both ranks.
  num_writers = 1 if rank == 0 else 0
  loader = MultiprocessShuffledDataloader(
    TorchrunDataset(),
    DataloaderConfig(
      bs=4,
      shuffle_size=32,
      min_mixing=0.0,
      num_writers=num_writers,
      num_readers=1,
      local_rank=rank,
      global_rank=rank,
      local_world_size=world_size,
      global_world_size=world_size,
      queue_name=queue_name,
      total_writers=1,
      global_writer_offset=0 if rank == 0 else 1,
      local_writer_offset=0 if rank == 0 else 1,
    ),
  )
  try:
    dummy = loader.get_dummy_batch()
    batch = next(iter(loader))
    assert dummy[0]['num_workers'].eq(1).all(), f"expected 1 total writer, got {dummy[0]['num_workers'].tolist()}"
    assert batch[0]['x'].shape == (4,)
    dist.barrier()
  finally:
    if dist.is_initialized():
      dist.destroy_process_group()


if __name__ == '__main__':
  main()
