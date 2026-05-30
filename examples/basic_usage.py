import os
import time
import logging

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from gigashuffle import DataloaderConfig, MultiprocessShuffledDataloader


NUM_BATCHES = 20
logger = logging.getLogger('gigashuffle.example')


class WorkerInfoDataset(IterableDataset):
  def __init__(
    self,
    chunk_size: int = 64,
    image_shape: tuple[int, int, int] = (3, 32, 32),
    num_classes: int = 10,
    device: str = 'cpu',
  ) -> None:
    self.chunk_size = chunk_size
    self.image_shape = image_shape
    self.num_classes = num_classes
    self.device = device

  def __iter__(self):
    while True:
      time.sleep(1)
      worker_info = get_worker_info()
      worker_id = -1 if worker_info is None else worker_info.id
      num_workers = 1 if worker_info is None else worker_info.num_workers
      images = torch.randn(self.chunk_size, *self.image_shape, device=self.device)
      labels = torch.randint(0, self.num_classes, (self.chunk_size,), device=self.device)
      worker_ids = torch.full((self.chunk_size,), worker_id, dtype=torch.int64, device=self.device)
      worker_count = torch.full((self.chunk_size,), num_workers, dtype=torch.int64, device=self.device)
      yield [{'images': images, 'labels': labels, 'worker_id': worker_ids, 'num_workers': worker_count}]


def main() -> None:
  dist.init_process_group(backend='gloo')

  local_rank = int(os.environ['LOCAL_RANK'])
  global_rank = int(os.environ['RANK'])
  local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
  global_world_size = int(os.environ['WORLD_SIZE'])
  node_rank = os.environ.get('GROUP_RANK', os.environ.get('NODE_RANK', '0'))
  redis_host = os.environ.get('REDIS_HOST', 'localhost')
  redis_port = int(os.environ.get('REDIS_PORT', '6379'))
  redis_db = int(os.environ.get('REDIS_DB', '6'))
  queue_prefix = os.environ.get('GIGASHUFFLE_QUEUE_PREFIX', f'basic-usage-node-{node_rank}')
  device = 'cpu'

  train_loader = MultiprocessShuffledDataloader(
    WorkerInfoDataset(chunk_size=16, device=device),
    DataloaderConfig(
      bs=16,
      shuffle_size=1024,
      num_writers=1,
      num_readers=1,
      local_rank=local_rank,
      global_rank=global_rank,
      local_world_size=local_world_size,
      global_world_size=global_world_size,
      redis_host=redis_host,
      redis_port=redis_port,
      redis_db=redis_db,
      queue_name=f'{queue_prefix}-train',
    ),
  )
  val_loader = MultiprocessShuffledDataloader(
    WorkerInfoDataset(chunk_size=16, device=device),
    DataloaderConfig(
      bs=16,
      shuffle_size=1024,
      num_writers=1,
      num_readers=1,
      local_rank=local_rank,
      global_rank=global_rank,
      local_world_size=local_world_size,
      global_world_size=global_world_size,
      redis_host=redis_host,
      redis_port=redis_port,
      redis_db=redis_db,
      queue_name=f'{queue_prefix}-val',
    ),
  )

  dummy_batch = train_loader.get_dummy_batch()
  logger.info("dummy train batch: images %s labels %s", tuple(dummy_batch[0]['images'].shape), tuple(dummy_batch[0]['labels'].shape))

  try:
    for name, loader, n_batches in [('train', train_loader, NUM_BATCHES), ('val', val_loader, 5)]:
      for i, batch in zip(range(n_batches), loader):
        images = batch[0]['images']
        labels = batch[0]['labels']
        worker_ids = sorted(batch[0]['worker_id'].unique().tolist())
        num_workers = sorted(batch[0]['num_workers'].unique().tolist())
        logger.info(
          "%s rank %d local_rank %d batch %d: images %s labels %s worker_ids %s num_workers %s",
          name,
          global_rank,
          local_rank,
          i,
          tuple(images.shape),
          tuple(labels.shape),
          worker_ids,
          num_workers,
        )

    dist.barrier()
  finally:
    train_loader.close()
    val_loader.close()
    if dist.is_initialized():
      dist.destroy_process_group()


if __name__ == "__main__":
  main()
