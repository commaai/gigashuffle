import os
import pickle
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from redis import StrictRedis
from torch.utils.data import IterableDataset, get_worker_info

from gigashuffle import DataloaderConfig, INDEX_KEY, MultiprocessShuffledDataloader, ShuffleBufferStats
from gigashuffle.multiprocess import BatchSizeMismatch, fetch_initial_sample, get_samples, write_samples_to_buffer


REDIS = dict(host=os.environ.get('REDIS_HOST', 'localhost'), port=int(os.environ.get('REDIS_PORT', '6379')), db=int(os.environ.get('REDIS_DB', '6')))


def torch_shm_names() -> set[str]:
  return {path.name for path in Path('/dev/shm').glob('torch_*')}


class RedisDataset(IterableDataset):
  def __init__(self, key: str, sleep: float = 0.0, device: str = 'cpu', x_offset: int = 0) -> None:
    self.key = key
    self.sleep = sleep
    self.device = device
    self.x_offset = x_offset

  def __iter__(self):
    r = StrictRedis(**REDIS)
    worker_info = get_worker_info()
    worker_id = -1 if worker_info is None else worker_info.id
    num_workers = 1 if worker_info is None else worker_info.num_workers
    r.incr(f'gigashuffle-{self.key}:iter:{worker_id}')
    i = 0
    while True:
      if i and self.sleep:
        time.sleep(self.sleep)
      x = torch.arange(4, device=self.device) + self.x_offset
      if self.device == 'cuda':
        x = (x * 2).cpu()
      r.incr(f'gigashuffle-{self.key}:samples:{worker_id}')
      yield [{'x': x, 'worker_id': torch.full((4,), worker_id), 'num_workers': torch.full((4,), num_workers)}]
      i += 1


class VariableBatchDataset(IterableDataset):
  def __iter__(self):
    sizes = [3, 5, 2, 6]
    i = 0
    while True:
      n = sizes[i % len(sizes)]
      yield [{'x': torch.arange(n), 'y': torch.full((n,), n)}]
      i += 1


class OrderedDataset(IterableDataset):
  def __iter__(self):
    i = 0
    while True:
      start = i * 4
      yield [{'x': torch.arange(start, start + 4)}]
      i += 1


@dataclass(frozen=True)
class TrainingContext:
  epoch: int
  step: int
  device: torch.device


class TrainingContextDataset(IterableDataset):
  def __init__(self, key: str) -> None:
    self.key = key
    self.context = None

  def __iter__(self):
    while True:
      epoch = -1 if self.context is None else self.context.epoch
      yield [{'epoch': torch.full((1,), epoch)}]


class SlowFirstSampleDataset(IterableDataset):
  def __init__(self, sleep: float) -> None:
    self.sleep = sleep

  def __iter__(self):
    time.sleep(self.sleep)
    while True:
      yield [{'x': torch.arange(4)}]


class FailingFirstSampleDataset(IterableDataset):
  def __iter__(self):
    raise RuntimeError("intentional dataset failure")


def config(queue_name: str, **kwargs) -> DataloaderConfig:
  opts = dict(bs=4, shuffle_size=32, min_mixing=0.0, num_writers=2, num_readers=2, redis_host=REDIS['host'], redis_port=REDIS['port'], redis_db=REDIS['db'], queue_name=queue_name)
  opts.update(kwargs)
  return DataloaderConfig(**opts)


def test_torchrun_gloo():
  env = os.environ.copy()
  env['GIGASHUFFLE_QUEUE'] = f'torchrun-{uuid.uuid4().hex}'
  cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=2', str(Path(__file__).with_name('torchrun_gloo.py'))]
  result = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=90)
  assert result.returncode == 0, result.stdout + result.stderr


def test_dummy_batch_returns_before_min_mixing():
  r = StrictRedis(**REDIS)
  queue_name = f'dummy-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(RedisDataset(queue_name, sleep=1.0), config(queue_name, shuffle_size=64, min_mixing=0.5))
  try:
    batch = loader.get_dummy_batch()
    assert int(r.scard(f'gigashuffle-{queue_name}-full')) < 32
    assert batch[0]['worker_id'].eq(0).all()
  finally:
    loader._shutdown_workers()


def test_evict_on_read_false_keeps_indices_until_explicit_evict():
  r = StrictRedis(**REDIS)
  queue_name = f'manual-evict-{uuid.uuid4().hex}'
  # sleep 10 to prevent racing
  loader = MultiprocessShuffledDataloader(RedisDataset(queue_name, sleep=10.0), config(queue_name, shuffle_size=12, num_writers=1, num_readers=1, evict_on_read=False))
  it = iter(loader)
  try:
    batch = next(it)
    indices = batch[0][INDEX_KEY].tolist()
    assert len(indices) == 4
    assert len(set(indices)) == 4

    full_key = f'gigashuffle-{queue_name}-full'
    empty_key = f'gigashuffle-{queue_name}-empty'
    assert set(indices) <= {int(x) for x in r.smembers(full_key)}
    empty_before = {int(x) for x in r.smembers(empty_key)}
    assert loader.evict(indices) == 4
    empty_after = {int(x) for x in r.smembers(empty_key)}
    assert set(indices).isdisjoint({int(x) for x in r.smembers(full_key)})
    assert empty_after == empty_before | set(indices)
  finally:
    del it
    loader._shutdown_workers()


def test_get_dummy_batch_aborts_when_writer_dies_before_attach_server():
  queue_name = f'dummy-dead-writer-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(FailingFirstSampleDataset(), config(queue_name, num_writers=1, num_readers=0))
  try:
    with pytest.raises(RuntimeError, match="child .* died"):
      loader.get_dummy_batch()
  finally:
    loader._shutdown_workers()


def test_stats_report_buffer_counts():
  r = StrictRedis(**REDIS)
  queue_name = f'stats-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(RedisDataset(queue_name, sleep=1.0), config(queue_name, shuffle_size=64, min_mixing=0.5))
  try:
    stats = loader.stats()
    assert isinstance(stats, ShuffleBufferStats)
    assert stats.full == int(r.scard(f'gigashuffle-{queue_name}-full'))
    assert stats.empty == int(r.scard(f'gigashuffle-{queue_name}-empty'))
    assert stats.in_flight == 64 - stats.full - stats.empty
  finally:
    loader._shutdown_workers()


def test_init_does_not_wait_for_metadata():
  queue_name = f'lazy-init-{uuid.uuid4().hex}'
  start = time.perf_counter()
  loader = MultiprocessShuffledDataloader(SlowFirstSampleDataset(sleep=5.0), config(queue_name, num_writers=1, num_readers=0))
  try:
    assert time.perf_counter() - start < 2.0
  finally:
    loader._shutdown_workers()


def test_fill_once_iter_does_not_wait_for_full_buffer():
  queue_name = f'lazy-fill-once-iter-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(SlowFirstSampleDataset(sleep=5.0), config(queue_name, shuffle_size=12, min_mixing=1, fill_once=True, num_writers=1, num_readers=1))
  try:
    start = time.perf_counter()
    iter(loader)
    assert time.perf_counter() - start < 2.0
  finally:
    loader._shutdown_workers()


def test_batch_size_mismatch():
  samples = iter([[{'x': torch.arange(2), 'y': torch.arange(3)}]])
  with pytest.raises(BatchSizeMismatch):
    get_samples(samples)


def test_writer_max_retries():
  writer_max_retries = 3
  samples = iter([[{'x': torch.empty(0)}] for _ in range(writer_max_retries)] + [[{'x': torch.arange(1)}]])
  with pytest.raises(ValueError, match=str(writer_max_retries)):
    fetch_initial_sample(samples, config(f'retries-{uuid.uuid4().hex}', writer_max_retries=writer_max_retries))


def test_loader_forces_fd_sharing_without_visible_torch_files():
  previous_strategy = torch.multiprocessing.get_sharing_strategy()
  before = torch_shm_names()
  torch.multiprocessing.set_sharing_strategy('file_system')
  queue_name = f'fd-shm-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(RedisDataset(queue_name), config(queue_name, num_writers=1, num_readers=1))
  try:
    assert loader.get_dummy_batch()[0]['x'].shape == (4,)
    assert next(iter(loader))[0]['x'].shape == (4,)
  finally:
    loader._shutdown_workers()
    torch.multiprocessing.set_sharing_strategy(previous_strategy)

  deadline = time.perf_counter() + 3
  while time.perf_counter() < deadline and torch_shm_names() - before:
    time.sleep(0.05)
  assert torch_shm_names() - before == set()


def test_different_input_batch_sizes():
  queue_name = f'sizes-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(VariableBatchDataset(), config(queue_name))
  try:
    batch = next(iter(loader))
    assert batch[0]['x'].shape == batch[0]['y'].shape == (4,)
  finally:
    loader._shutdown_workers()


def test_write_samples_slices_to_acquired_slots():
  shuffle_buffer = [{'x': torch.full((3,), -1, dtype=torch.int64)}]
  samples = [{'x': torch.arange(5)}]

  write_samples_to_buffer(shuffle_buffer, samples, [1, 2], local_input_bs=2)

  assert shuffle_buffer[0]['x'].tolist() == [-1, 0, 1]


def test_fill_once_loops_in_order():
  r = StrictRedis(**REDIS)
  queue_name = f'fill-once-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(OrderedDataset(), config(queue_name, shuffle_size=12, min_mixing=1, fill_once=True, num_readers=1, num_writers=1))
  try:
    assert loader.get_dummy_batch()[0]['x'].tolist() == [0, 1, 2, 3]
    deadline = time.perf_counter() + 5
    while time.perf_counter() < deadline and int(r.scard(f'gigashuffle-{queue_name}-full')) < 12:
      time.sleep(0.05)
    assert int(r.scard(f'gigashuffle-{queue_name}-full')) == 12
    it = iter(loader)
    batches = [next(it)[0]['x'].tolist() for _ in range(3)]
    assert batches[0] == [0, 1, 2, 3]
    assert sorted(x for batch in batches for x in batch) == list(range(12))
    with pytest.raises(StopIteration):
      next(it)
    assert int(r.scard(f'gigashuffle-{queue_name}-empty')) == 0
    assert len(loader.children) == 1
    writer = loader.children[0]
    assert writer.is_alive()
    assert loader.get_dummy_batch()[0]['x'].tolist() == [0, 1, 2, 3]
    loader.check_children()
  finally:
    loader._shutdown_workers()


def test_fill_once_iters_repeat():
  queue_name = f'fill-once-repeat-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(OrderedDataset(), config(queue_name, shuffle_size=12, min_mixing=1, fill_once=True, num_readers=1))
  try:
    first = [batch[0]['x'].tolist() for batch in loader]
    second = [batch[0]['x'].tolist() for batch in loader]
    assert first == second
  finally:
    loader._shutdown_workers()


def test_fill_once_iters_repeat_underconsumed():
  queue_name = f'fill-once-repeat-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(OrderedDataset(), config(queue_name, shuffle_size=24, min_mixing=1, fill_once=True, num_readers=1))
  try:
    ll = iter(loader)
    first = [next(ll)[0]['x'].tolist() for _ in range(3)]
    ll = iter(loader)
    second = [next(ll)[0]['x'].tolist() for _ in range(3)]
    assert first == second
  finally:
    loader._shutdown_workers()


def test_fill_once_getitem_reads_absolute_buffer_index():
  queue_name = f'fill-once-getitem-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(OrderedDataset(), config(queue_name, shuffle_size=12, min_mixing=1, fill_once=True, num_readers=1, num_writers=1))
  try:
    first_item = loader[0]
    assert first_item[0]['x'].shape == torch.Size([])
    assert first_item[0]['x'].item() == 0
    assert first_item[0][INDEX_KEY].shape == torch.Size([])
    assert first_item[0][INDEX_KEY].item() == 0

    by_idx = {}
    for batch in loader:
      xs = batch[0]['x'].clone()
      idxs = batch[0][INDEX_KEY].clone()
      for idx, x in zip(idxs.tolist(), xs):
        by_idx[idx] = x.clone()

    assert set(by_idx) == set(range(12))
    for idx in [0, 5, 11]:
      item = loader[idx]
      assert item[0]['x'].shape == torch.Size([])
      assert torch.equal(item[0]['x'], by_idx[idx])
      assert item[0][INDEX_KEY].shape == torch.Size([])
      assert item[0][INDEX_KEY].item() == idx

    with pytest.raises(IndexError):
      loader[-1]
    with pytest.raises(IndexError):
      loader[12]
  finally:
    loader._shutdown_workers()


def test_getitem_requires_fill_once():
  queue_name = f'getitem-streaming-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(RedisDataset(queue_name), config(queue_name, num_writers=1, num_readers=1))
  try:
    with pytest.raises(RuntimeError, match='fill_once=True'):
      loader[0]
  finally:
    loader._shutdown_workers()


def test_multiple_loaders_fill_together():
  q1, q2 = f'train-{uuid.uuid4().hex}', f'val-{uuid.uuid4().hex}'
  loader1 = MultiprocessShuffledDataloader(RedisDataset(q1), config(q1))
  loader2 = MultiprocessShuffledDataloader(RedisDataset(q2, x_offset=100), config(q2))
  try:
    dummy1 = loader1.get_dummy_batch()
    dummy2 = loader2.get_dummy_batch()
    batch1 = next(iter(loader1))
    batch2 = next(iter(loader2))
    assert dummy1[0]['num_workers'].eq(2).all()
    assert dummy2[0]['num_workers'].eq(2).all()
    assert batch1[0]['x'].shape == (4,)
    assert batch2[0]['x'].shape == (4,)
    assert not torch.equal(dummy1[0]['x'], dummy2[0]['x'])
    assert not torch.equal(batch1[0]['x'], batch2[0]['x'])
  finally:
    loader1._shutdown_workers()
    loader2._shutdown_workers()


def test_worker_info_and_iter_once():
  r = StrictRedis(**REDIS)
  queue_name = f'workers-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(RedisDataset(queue_name), config(queue_name, num_writers=2))
  try:
    batch = next(iter(loader))
    assert set(batch[0]['num_workers'].tolist()) == {2}
    deadline = time.perf_counter() + 5
    while time.perf_counter() < deadline and int(r.get(f'gigashuffle-{queue_name}:samples:1') or 0) == 0:
      time.sleep(0.05)
    assert [int(r.get(f'gigashuffle-{queue_name}:iter:{i}') or 0) for i in range(2)] == [1, 1]
  finally:
    loader._shutdown_workers()


def test_attach_training_context_updates_writer_dataset_epoch():
  r = StrictRedis(**REDIS)
  queue_name = f'context-{uuid.uuid4().hex}'
  context = TrainingContext(epoch=7, step=3, device=torch.device('cpu'))
  r.set(f'gigashuffle-{queue_name}-training-context-global_rank_0', pickle.dumps(context))
  loader = MultiprocessShuffledDataloader(TrainingContextDataset(queue_name), config(queue_name, bs=1, shuffle_size=3, num_writers=1, num_readers=1))
  iloader = iter(loader)
  try:
    batch = next(iloader)
    loader.attach_training_context(context)
    assert batch[0]['epoch'].item() == -1
    for _ in range(10):
      batch = next(iloader)
    assert batch[0]['epoch'].item() == 7
  finally:
    del iloader
    loader._shutdown_workers()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda unavailable')
def test_cuda_work_inside_worker():
  queue_name = f'cuda-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(RedisDataset(queue_name, device='cuda'), config(queue_name))
  try:
    batch = loader.get_dummy_batch()
    assert batch[0]['x'].tolist() == [0, 2, 4, 6]
  finally:
    loader._shutdown_workers()


def test_check_children_health():
  queue_name = f'health-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(RedisDataset(queue_name), config(queue_name))
  try:
    loader.children[0].terminate()
    loader.children[0].join(timeout=5)
    with pytest.raises(RuntimeError, match='child'):
      loader.check_children()
  finally:
    loader._shutdown_workers()
