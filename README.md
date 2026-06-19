# gigashuffle

Shuffle-buffer dataloading for PyTorch training.

## Usage

Start a redis server before constructing a loader:

```bash
redis-server
```

Use in your training script:

```python
import os
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from gigashuffle import DataloaderConfig, MultiprocessShuffledDataloader

class MyDataset(IterableDataset):
  def __iter__(self):
    while True:
      x = torch.randn(64, 3, 224, 224)
      y = torch.randint(0, 1000, (64,))
      yield [{'x': x, 'y': y}]

config = DataloaderConfig(
  bs=32,
  shuffle_size=1000,
  num_writers=2,
  num_readers=2,
  local_rank=int(os.environ.get('LOCAL_RANK', '0')),
  global_rank=int(os.environ.get('RANK', '0')),
  local_world_size=int(os.environ.get('LOCAL_WORLD_SIZE', '1')),
  global_world_size=int(os.environ.get('WORLD_SIZE', '1')),
  queue_name='my-training-loader', # must be unique across your dataloaders
  fill_once=False, # see below
)

loader = MultiprocessShuffledDataloader(MyDataset(), config=config)
dummy_batch = loader.get_dummy_batch()

for batch in loader:
  x = batch[0]['x']
  y = batch[0]['y']
  break
```

Datasets yield a `Buffer`: `list[dict[str, Tensor | ndarray]]`. Every tensor or array in a sample must share the same first dimension; that is the input batch size. `config.bs` is the output batch size.

Call `loader.get_dummy_batch()` when you need a sanity-check batch before the shuffle buffer reaches `min_mixing`. It repeats the initial sample to `config.bs` and does not advance the normal shuffled iterator.

Set `config.fill_once=True`, `config.min_mixing=1`, and `config.num_readers=1` to populate the shuffle buffer once, then yield one ordered pass over it without returning indices to the writers. Reader waits for the full `shuffle_size` before yielding batches in this mode.

By default (`config.evict_on_read=True`) a buffer index is returned to the writers as soon as it is read, so each sample is consumed once. Set `config.evict_on_read=False` to keep read entries in the buffer for re-sampling and have the client manage eviction with `loader.evict(indices)`. Never evicting will freeze the buffer once it fills, and evicting faster than writers can refill will make reads wait. Every batch carries the buffer index of each row in `batch[0][INDEX_KEY]` (a `(bs,)` int64 tensor). `config.evict_on_read=False` is not supported with `fill_once=True`.

## Notes

Each Dataloader owns one shared CPU shuffle buffer. The owner writer (`proc_idx=0`) allocates the buffer with `Tensor.share_memory_()` using PyTorch's `file_descriptor` CPU sharing strategy, publishes only an `AF_UNIX` attach-socket path in Redis, and sends the shared tensor objects over that socket; this follows the same fd-transfer mechanism PyTorch uses for CPU tensor IPC. The simpler alternatives do not work well here: a `multiprocessing.Queue` object cannot be shared across independent `torchrun` ranks, Redis cannot transmit process-local file descriptors, explicit `/dev/shm` or `torch.from_file` names can leak after killed workers, and `/proc/<pid>/fd/<fd>` attachment is Linux/container-permission fragile and races owner death.

The output batches are views into reusable shared-memory reader buffers, not copies. Their contents may be overwritten after the iterator advances or the batch is released, so callers that need to keep a batch must clone/copy it before requesting another batch or dropping the iterator.
