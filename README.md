# gigashuffle

Shuffle-buffer dataloading for PyTorch training.

Gigashuffle continuously streams samples through a large named shared-memory buffer and returns random batches from that buffer. It is built for `torchrun`: sibling ranks on a node coordinate through Redis, local rank 0 creates the `/dev/shm` tensors, and the other local ranks attach to that buffer.

## Install

```bash
uv sync
```

Or install into an existing environment:

```bash
pip install -e .
```

Start Redis yourself before constructing a loader:

```bash
redis-server
```

## Example

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 examples/basic_usage.py
```

The example intentionally sleeps while producing samples so `get_dummy_batch()` returns before the shuffle buffer reaches `min_mixing`. Set `GIGASHUFFLE_SAMPLE_SLEEP_S=0` to make it fast again.

## Usage

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

dist.init_process_group(backend='gloo')

config = DataloaderConfig(
  bs=32,
  shuffle_size=10000,
  num_writers=2,
  num_readers=2,
  local_rank=int(os.environ['LOCAL_RANK']),
  global_rank=int(os.environ['RANK']),
  local_world_size=int(os.environ.get('LOCAL_WORLD_SIZE', '1')),
  global_world_size=int(os.environ['WORLD_SIZE']),
  queue_name='my-training-loader',
)

loader = MultiprocessShuffledDataloader(MyDataset(), config=config)
dummy_batch = loader.get_dummy_batch()

for batch in loader:
  x = batch[0]['x']
  y = batch[0]['y']
  break

loader.close()
```

Datasets yield a `Buffer`: `list[dict[str, Tensor | ndarray]]`. Every tensor or array in a sample must share the same first dimension; that is the input chunk size. `config.bs` is the output batch size.

Call `loader.get_dummy_batch()` when you need a sanity-check batch before the shuffle buffer reaches `min_mixing`. It repeats the initial sample to `config.bs` and does not advance the normal shuffled iterator.

Inside writer processes, `torch.utils.data.get_worker_info()` is populated with `id`, `num_workers`, `seed`, and `dataset`, so iterable datasets can shard or seed themselves the same way they would under native PyTorch workers.

`queue_name` is required. Use a different name for every live local loader, for example one name for train and one name for each validation set. On multi-node jobs, include the node rank or hostname in the name because `/dev/shm` is local to one machine.
