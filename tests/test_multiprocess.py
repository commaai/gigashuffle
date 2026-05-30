import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
import torch
from redis import StrictRedis
from torch.utils.data import IterableDataset, get_worker_info

from gigashuffle import DataloaderConfig, MultiprocessShuffledDataloader, ShuffleBufferStats
from gigashuffle.multiprocess import BatchSizeMismatch, FILL_ONCE_WRITER_DONE_EXITCODE, attach_named_shuffle_buffer, fetch_initial_sample, get_samples, initialize_redis_queue


REDIS = dict(host=os.environ.get('REDIS_HOST', 'localhost'), port=int(os.environ.get('REDIS_PORT', '6379')), db=int(os.environ.get('REDIS_DB', '6')))


class RedisDataset(IterableDataset):
  def __init__(self, key: str, sleep: float = 0.0, device: str = 'cpu') -> None:
    self.key = key
    self.sleep = sleep
    self.device = device

  def __iter__(self):
    r = StrictRedis(**REDIS)
    worker_info = get_worker_info()
    worker_id = -1 if worker_info is None else worker_info.id
    num_workers = 1 if worker_info is None else worker_info.num_workers
    r.incr(f'{self.key}:iter:{worker_id}')
    i = 0
    while True:
      if i and self.sleep:
        time.sleep(self.sleep)
      x = torch.arange(4, device=self.device)
      if self.device == 'cuda':
        x = (x * 2).cpu()
      r.incr(f'{self.key}:samples:{worker_id}')
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


def config(queue_name: str, shm_dir: Path, **kwargs) -> DataloaderConfig:
  opts = dict(bs=4, shuffle_size=32, min_mixing=0.0, num_writers=2, num_readers=2, redis_host=REDIS['host'], redis_port=REDIS['port'], redis_db=REDIS['db'], shm_dir=str(shm_dir), queue_name=queue_name)
  opts.update(kwargs)
  return DataloaderConfig(**opts)


def test_torchrun_gloo(tmp_path):
  env = os.environ.copy()
  env['GIGASHUFFLE_QUEUE'] = f'torchrun-{uuid.uuid4().hex}'
  env['GIGASHUFFLE_SHM_DIR'] = str(tmp_path)
  cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=2', str(Path(__file__).with_name('torchrun_gloo.py'))]
  result = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=90)
  assert result.returncode == 0, result.stdout + result.stderr


def test_dummy_batch_returns_before_min_mixing(tmp_path):
  r = StrictRedis(**REDIS)
  queue_name = f'dummy-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(RedisDataset(queue_name, sleep=1.0), config(queue_name, tmp_path, shuffle_size=64, min_mixing=0.5))
  try:
    batch = loader.get_dummy_batch()
    assert int(r.scard(f'{queue_name}-full')) < 32
    assert batch[0]['worker_id'].eq(0).all()
  finally:
    loader.close()


def test_stats_report_buffer_counts(tmp_path):
  r = StrictRedis(**REDIS)
  queue_name = f'stats-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(RedisDataset(queue_name, sleep=1.0), config(queue_name, tmp_path, shuffle_size=64, min_mixing=0.5))
  try:
    stats = loader.stats()
    assert isinstance(stats, ShuffleBufferStats)
    assert stats.full == int(r.scard(f'{queue_name}-full'))
    assert stats.empty == int(r.scard(f'{queue_name}-empty'))
    assert stats.in_flight == 64 - stats.full - stats.empty
  finally:
    loader.close()


def test_batch_size_mismatch():
  samples = iter([[{'x': torch.arange(2), 'y': torch.arange(3)}]])
  with pytest.raises(BatchSizeMismatch):
    get_samples(samples)


def test_writer_max_retries(tmp_path):
  writer_max_retries = 3
  samples = iter([[{'x': torch.empty(0)}] for _ in range(writer_max_retries)] + [[{'x': torch.arange(1)}]])
  with pytest.raises(ValueError, match=str(writer_max_retries)):
    fetch_initial_sample(samples, config(f'retries-{uuid.uuid4().hex}', tmp_path, writer_max_retries=writer_max_retries))


def test_initialize_redis_queue_clears_metadata():
  r = StrictRedis(**REDIS)
  queue_name = f'init-{uuid.uuid4().hex}'
  r.set(f'{queue_name}-shared-buffer-meta', b'stale')
  initialize_redis_queue(r, queue_name, 4)
  assert r.get(f'{queue_name}-shared-buffer-meta') is None
  assert int(r.scard(f'{queue_name}-empty')) == 4
  assert int(r.scard(f'{queue_name}-full')) == 0


def test_different_input_batch_sizes(tmp_path):
  queue_name = f'sizes-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(VariableBatchDataset(), config(queue_name, tmp_path))
  try:
    batch = next(iter(loader))
    assert batch[0]['x'].shape == batch[0]['y'].shape == (4,)
  finally:
    loader.close()


def test_fill_once_loops_in_order(tmp_path):
  r = StrictRedis(**REDIS)
  queue_name = f'fill-once-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(OrderedDataset(), config(queue_name, tmp_path, shuffle_size=12, min_mixing=1, fill_once=True, num_readers=1))
  try:
    deadline = time.perf_counter() + 5
    while time.perf_counter() < deadline and int(r.scard(f'{queue_name}-full')) < 12:
      time.sleep(0.05)
    assert int(r.scard(f'{queue_name}-full')) == 12
    shuffle_buffer = attach_named_shuffle_buffer(loader.shuffle_buffer_metadata)
    expected = [shuffle_buffer[0]['x'][0:4].tolist(), shuffle_buffer[0]['x'][4:8].tolist(), shuffle_buffer[0]['x'][8:12].tolist()]
    it = iter(loader)
    assert [next(it)[0]['x'].tolist() for _ in range(3)] == expected
    with pytest.raises(StopIteration):
      next(it)
    assert int(r.scard(f'{queue_name}-empty')) == 0
    writer = loader.children[1]
    deadline = time.perf_counter() + 5
    while time.perf_counter() < deadline and writer.exitcode is None:
      writer.join(timeout=0.05)
    assert writer.exitcode == FILL_ONCE_WRITER_DONE_EXITCODE
    loader.check_children()
  finally:
    loader.close()


def test_fill_once_iters_repeat(tmp_path):
  queue_name = f'fill-once-repeat-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(OrderedDataset(), config(queue_name, tmp_path, shuffle_size=12, min_mixing=1, fill_once=True, num_readers=1))
  try:
    first = [batch[0]['x'].tolist() for batch in loader]
    second = [batch[0]['x'].tolist() for batch in loader]
    assert first == second
  finally:
    loader.close()


def test_multiple_loaders_fill_together(tmp_path):
  r = StrictRedis(**REDIS)
  q1, q2 = f'train-{uuid.uuid4().hex}', f'val-{uuid.uuid4().hex}'
  loader1 = MultiprocessShuffledDataloader(RedisDataset(q1, sleep=0.05), config(q1, tmp_path / 'train'))
  loader2 = MultiprocessShuffledDataloader(RedisDataset(q2, sleep=0.05), config(q2, tmp_path / 'val'))
  try:
    deadline = time.perf_counter() + 5
    while time.perf_counter() < deadline:
      n1 = int(r.get(f'{q1}:samples:0') or 0)
      n2 = int(r.get(f'{q2}:samples:0') or 0)
      if n1 > 1 and n2 > 1:
        break
      time.sleep(0.05)
    assert n1 > 1 and n2 > 1
  finally:
    loader1.close()
    loader2.close()


def test_worker_info_and_iter_once(tmp_path):
  r = StrictRedis(**REDIS)
  queue_name = f'workers-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(RedisDataset(queue_name), config(queue_name, tmp_path, num_writers=2))
  try:
    batch = next(iter(loader))
    assert set(batch[0]['num_workers'].tolist()) == {2}
    deadline = time.perf_counter() + 5
    while time.perf_counter() < deadline and int(r.get(f'{queue_name}:samples:1') or 0) == 0:
      time.sleep(0.05)
    assert [int(r.get(f'{queue_name}:iter:{i}') or 0) for i in range(2)] == [1, 1]
  finally:
    loader.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda unavailable')
def test_cuda_work_inside_worker(tmp_path):
  queue_name = f'cuda-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(RedisDataset(queue_name, device='cuda'), config(queue_name, tmp_path))
  try:
    batch = loader.get_dummy_batch()
    assert batch[0]['x'].tolist() == [0, 2, 4, 6]
  finally:
    loader.close()


def test_check_children_health(tmp_path):
  queue_name = f'health-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(RedisDataset(queue_name), config(queue_name, tmp_path))
  try:
    loader.children[0].terminate()
    loader.children[0].join(timeout=5)
    with pytest.raises(RuntimeError, match='child'):
      loader.check_children()
  finally:
    loader.close()
