import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch.utils.data import IterableDataset, get_worker_info

import gigashuffle.stats as stats_cli
from gigashuffle import DataloaderConfig, INDEX_KEY, MultiprocessShuffledDataloader, ShuffleBufferStats
from gigashuffle.multiprocess import (
  BatchSizeMismatch,
  fetch_initial_sample,
  get_samples,
  write_samples_to_buffer,
)


def torch_shm_names() -> set[str]:
  return {path.name for path in Path('/dev/shm').glob('torch_*')}


class ToyDataset(IterableDataset):
  def __init__(self, sleep: float = 0.0, device: str = 'cpu', x_offset: int = 0) -> None:
    self.sleep = sleep
    self.device = device
    self.x_offset = x_offset

  def __iter__(self):
    worker_info = get_worker_info()
    worker_id = -1 if worker_info is None else worker_info.id
    num_workers = 1 if worker_info is None else worker_info.num_workers
    i = 0
    while True:
      if i and self.sleep:
        time.sleep(self.sleep)
      x = torch.arange(4, device=self.device) + self.x_offset
      if self.device == 'cuda':
        x = (x * 2).cpu()
      yield [{'x': x, 'worker_id': torch.full((4,), worker_id), 'num_workers': torch.full((4,), num_workers)}]
      i += 1


class CountingDataset(IterableDataset):
  def __init__(self, counter_dir: Path) -> None:
    self.counter_dir = counter_dir

  def __iter__(self):
    self.counter_dir.mkdir(parents=True, exist_ok=True)
    worker_info = get_worker_info()
    worker_id = -1 if worker_info is None else worker_info.id
    num_workers = 1 if worker_info is None else worker_info.num_workers
    (self.counter_dir / f'iter-{worker_id}').write_text('1')
    i = 0
    while True:
      (self.counter_dir / f'samples-{worker_id}').write_text(str(i + 1))
      yield [{'x': torch.arange(4), 'worker_id': torch.full((4,), worker_id), 'num_workers': torch.full((4,), num_workers)}]
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
  def __init__(self) -> None:
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
  opts = dict(bs=4, shuffle_size=32, min_mixing=0.0, num_writers=2, num_readers=2, queue_name=queue_name)
  opts.update(kwargs)
  return DataloaderConfig(**opts)


def test_torchrun_gloo():
  env = os.environ.copy()
  env['GIGASHUFFLE_QUEUE'] = f'torchrun-{uuid.uuid4().hex}'
  cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=2', str(Path(__file__).with_name('torchrun_gloo.py'))]
  result = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=90)
  assert result.returncode == 0, result.stdout + result.stderr


def test_dummy_batch_returns_before_min_mixing():
  queue_name = f'dummy-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(ToyDataset(sleep=1.0), config(queue_name, shuffle_size=64, min_mixing=0.5))
  try:
    batch = loader.get_dummy_batch()
    assert loader.stats().full < 32
    assert batch[0]['worker_id'].eq(0).all()
  finally:
    loader._shutdown_workers()


def test_evict_on_read_false_keeps_indices_until_explicit_evict():
  queue_name = f'manual-evict-{uuid.uuid4().hex}'
  # sleep 10 to prevent racing
  loader = MultiprocessShuffledDataloader(ToyDataset(sleep=10.0), config(queue_name, shuffle_size=12, num_writers=1, num_readers=1, evict_on_read=False))
  it = iter(loader)
  try:
    batch = next(it)
    indices = batch[0][INDEX_KEY].tolist()
    assert len(indices) == 4
    assert len(set(indices)) == 4

    stats_before = loader.stats()
    assert loader.evict(indices) == 4
    stats_after = loader.stats()
    assert stats_after.full == stats_before.full - 4
    assert stats_after.empty == stats_before.empty + 4
  finally:
    del it
    loader._shutdown_workers()


def test_get_dummy_batch_aborts_when_writer_dies_before_coordinator():
  queue_name = f'dummy-dead-writer-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(FailingFirstSampleDataset(), config(queue_name, num_writers=1, num_readers=0))
  try:
    with pytest.raises(RuntimeError, match="child .* died"):
      loader.get_dummy_batch()
  finally:
    loader._shutdown_workers()


def test_stats_report_buffer_counts():
  queue_name = f'stats-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(ToyDataset(sleep=1.0), config(queue_name, shuffle_size=64, min_mixing=0.5))
  try:
    stats = loader.stats()
    assert isinstance(stats, ShuffleBufferStats)
    assert stats.full >= 0
    assert stats.empty >= 0
    assert stats.in_flight >= 0
    assert stats.full + stats.empty + stats.in_flight == 64
  finally:
    loader._shutdown_workers()


def test_stats_cli_reports_live_coordinator(capsys):
  queue_name = f'stats-cli-{uuid.uuid4().hex}'
  full_queue_name = f'gigashuffle-{queue_name}'
  loader = MultiprocessShuffledDataloader(ToyDataset(sleep=1.0), config(queue_name, num_writers=1, num_readers=1))
  try:
    loader.get_dummy_batch()
    rows = [s for s in stats_cli.live_stats() if s['queue_name'] == full_queue_name]
    assert len(rows) == 1
    assert rows[0]['full'] + rows[0]['empty'] + rows[0]['in_flight'] == 32

    stats_cli.print_stats()
    out = capsys.readouterr().out
    assert "queue_name full empty in_flight attached" in out
    assert full_queue_name in out
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
  loader = MultiprocessShuffledDataloader(ToyDataset(), config(queue_name, num_writers=1, num_readers=1))
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
  queue_name = f'fill-once-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(OrderedDataset(), config(queue_name, shuffle_size=12, min_mixing=1, fill_once=True, num_readers=1, num_writers=1))
  try:
    assert loader.get_dummy_batch()[0]['x'].tolist() == [0, 1, 2, 3]
    deadline = time.perf_counter() + 5
    while time.perf_counter() < deadline and loader.stats().full < 12:
      time.sleep(0.05)
    assert loader.stats().full == 12
    it = iter(loader)
    batches = [next(it)[0]['x'].tolist() for _ in range(3)]
    assert batches[0] == [0, 1, 2, 3]
    assert sorted(x for batch in batches for x in batch) == list(range(12))
    with pytest.raises(StopIteration):
      next(it)
    assert loader.stats().empty == 0
    writer = loader.children[1]
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


def test_multiple_loaders_fill_together():
  q1, q2 = f'train-{uuid.uuid4().hex}', f'val-{uuid.uuid4().hex}'
  loader1 = MultiprocessShuffledDataloader(ToyDataset(), config(q1))
  loader2 = MultiprocessShuffledDataloader(ToyDataset(x_offset=100), config(q2))
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


def test_worker_info_and_iter_once(tmp_path):
  queue_name = f'workers-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(CountingDataset(tmp_path), config(queue_name, num_writers=2))
  try:
    batch = next(iter(loader))
    assert set(batch[0]['num_workers'].tolist()) == {2}
    deadline = time.perf_counter() + 5
    while time.perf_counter() < deadline and not (tmp_path / 'samples-1').exists():
      time.sleep(0.05)
    assert [(tmp_path / f'iter-{i}').exists() for i in range(2)] == [True, True]
  finally:
    loader._shutdown_workers()


def test_attach_training_context_updates_writer_dataset_epoch():
  queue_name = f'context-{uuid.uuid4().hex}'
  context = TrainingContext(epoch=7, step=3, device=torch.device('cpu'))
  loader = MultiprocessShuffledDataloader(TrainingContextDataset(), config(queue_name, bs=1, shuffle_size=3, num_writers=1, num_readers=1))
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
  loader = MultiprocessShuffledDataloader(ToyDataset(device='cuda'), config(queue_name))
  try:
    batch = loader.get_dummy_batch()
    assert batch[0]['x'].tolist() == [0, 2, 4, 6]
  finally:
    loader._shutdown_workers()


def test_check_children_health():
  queue_name = f'health-{uuid.uuid4().hex}'
  loader = MultiprocessShuffledDataloader(ToyDataset(), config(queue_name))
  try:
    loader.children[0].terminate()
    loader.children[0].join(timeout=5)
    with pytest.raises(RuntimeError, match='child'):
      loader.check_children()
  finally:
    loader._shutdown_workers()
