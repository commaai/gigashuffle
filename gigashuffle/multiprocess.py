import atexit
import ctypes
import logging
import os
import pickle
import random
import signal
import sys
import time
from dataclasses import dataclass
from itertools import count
from multiprocessing.synchronize import Event
from multiprocessing.queues import SimpleQueue
from typing import Any, Iterator, cast

import torch
import numpy as np
import torch.multiprocessing as mp
from setproctitle import setproctitle
from torch.utils.data import Dataset, IterableDataset
from gigashuffle.coordinator import COORDINATOR_CONNECT_ERRORS, CoordinatorClient, CoordinatorServer, coordinator_authkey
from gigashuffle.worker_info import set_worker_info
from gigashuffle.config import DataloaderConfig


Buffer = list[dict[str, torch.Tensor]]
INDEX_KEY = '_gigashuffle_idx'
RANK_ID_FORMAT = 'global_rank_{global_rank}'
LOG_INTERVAL_S = 5.0
PR_SET_PDEATHSIG = 1
FILL_ONCE_WRITER_DONE_EXITCODE = 81
CLOSE_JOIN_TIMEOUT_S = 0.2
ShuffleBufferMetadata = dict[str, Any]
logger = logging.getLogger(__name__)
already_warned = False
_python_exit_status = False


@dataclass(frozen=True, slots=True)
class ShuffleBufferAttachment:
  metadata: ShuffleBufferMetadata
  shuffle_buffer: Buffer
  dummy_batch: Buffer


def _set_python_exit_status() -> None:
  global _python_exit_status
  _python_exit_status = True


atexit.register(_set_python_exit_status)

# NOTE: For high-throughput tasks, calling torch.set_num_threads(1) seems to significantly reduce CPU usage in the gigashuffle writers.
# This SO thread about a similar issue notes that it "was caused by a bad interaction of OpenMP and multiprocessing".
# https://stackoverflow.com/questions/65057388/pytorch-multiprocessing-with-shared-memory-causes-matmul-to-be-30x-slower-with
if sys.platform == 'linux':
  mp.set_sharing_strategy('file_descriptor')
torch.set_num_threads(1)


class BatchSizeMismatch(Exception):
  pass


@dataclass(frozen=True, slots=True)
class ShuffleBufferStats:
  full: int
  empty: int
  in_flight: int


def init_logger() -> None:
  package_logger = logging.getLogger('gigashuffle')
  package_logger.setLevel(logging.INFO)
  package_logger.handlers.clear()
  package_logger.propagate = False
  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  ch.setFormatter(logging.Formatter("[gigashuffle] %(asctime)s - %(name)s - %(levelname)s - %(message)s"))
  package_logger.addHandler(ch)
  os.environ["KINETO_LOG_LEVEL"] = "5"


def get_elapsed_and_eta(now: float, start_time: float, current: int, start_current: int, target: int) -> tuple[float, float]:
  elapsed_s = now - start_time
  progress = current - start_current
  eta_s = (target - current) * elapsed_s / progress if progress > 0 else float('nan')
  return elapsed_s, eta_s


def initialize_shuffle_buffer_tensor_ipc() -> None:
  if sys.platform == 'linux':
    mp.set_sharing_strategy('file_descriptor')
  mp.current_process().authkey = coordinator_authkey()


def _prctl_pr_set_pdeathsig(signum: int) -> None:
  if sys.platform != 'linux':
    return
  libc = ctypes.CDLL(None, use_errno=True)
  result = libc.prctl(PR_SET_PDEATHSIG, signum, 0, 0, 0)
  if result != 0:
    errno = ctypes.get_errno()
    raise OSError(errno, os.strerror(errno))


def set_parent_death_signal(parent_pid: int) -> None:
  _prctl_pr_set_pdeathsig(signal.SIGKILL)
  if sys.platform == 'linux' and os.getppid() != parent_pid:
    os.kill(os.getpid(), signal.SIGKILL)


def print_small_shuffle_warning():
  global already_warned
  if not already_warned:
    logger.warning("Shuffle buffer is too small to fit a single input batch. The input batch will be truncated to fit, which can significantly degrade dataloader performance.")
    already_warned = True


def numpy_type_to_torch(x):
  if x == np.bool_ or x == torch.bool:
    return torch.bool
  elif x == np.uint8 or x == torch.uint8:
    return torch.uint8
  elif x == np.int16 or x == torch.int16:
    return torch.int16
  elif x == np.int64 or x == torch.int64:
    return torch.int64
  elif x == np.int32 or x == torch.int32:
    return torch.int32
  elif x == np.float32 or x == torch.float32:
    return torch.float32
  elif x == np.float16 or x == torch.float16:
    return torch.float16
  else:
    raise Exception(f"unsupported numpy type {x!r}")


def get_input_bs_key(samples):
  for i in range(len(samples)):
    for k in samples[i]:
      return i, k


def assert_bs_equal(samples, input_bs=None):
  for i in range(len(samples)):
    for k in samples[i]:
      if input_bs is None:
        input_bs = samples[i][k].shape[0]
      if input_bs != samples[i][k].shape[0]:
        raise BatchSizeMismatch("batch size mismatch in samples")


def get_samples(dset, input_bs_key=None, max_retries=100):
  for i in range(max_retries):
    samples = next(dset) if hasattr(dset, '__next__') else random.choice(dset)
    if input_bs_key is None:
      input_bs_key = get_input_bs_key(samples)
    input_bs = samples[input_bs_key[0]][input_bs_key[1]].shape[0]
    if input_bs > 0:
      assert_bs_equal(samples, input_bs)
      return samples, input_bs, input_bs_key
  raise ValueError(f"dataset returned only empty samples in {max_retries} attempts")


def get_memory_size(first_samples, input_bs):
  memory_size = 0
  for i in range(len(first_samples)):
    for v in first_samples[i].values():
      assert input_bs == v.shape[0]
      if isinstance(v, torch.Tensor):
        memory_size += v.untyped_storage().size()
      else:
        memory_size += v.nbytes
  return memory_size // input_bs


def fetch_initial_sample(dset: Any, config: DataloaderConfig) -> tuple[Buffer, int, tuple[int, str]]:
  shuffle_size = config.shuffle_size
  min_mixing_n = int(config.min_mixing * shuffle_size)
  input_samples, input_bs, input_bs_key = get_samples(dset, max_retries=config.writer_max_retries)
  memory_size = get_memory_size(input_samples, input_bs)

  logger.info(f"each element uses {memory_size} bytes, total buffer size is {shuffle_size*memory_size/1e9:.3f}gb")
  if shuffle_size < config.bs * config.local_world_size:
    N = config.local_world_size * config.num_readers
    raise RuntimeError(f"Shuffle buffer must be large enough to accommodate at least N batches, but buffer size = {shuffle_size}, batch size = {config.bs}, N = {N}")
  if not config.fill_once and min_mixing_n >= shuffle_size - 2*input_bs:
    raise RuntimeError(f"To avoid deadlock, min_mixing_n ({min_mixing_n}) must be less than {shuffle_size - 2*input_bs}")

  return input_samples, input_bs, input_bs_key


def fetch_rand_from_queue(coord: CoordinatorClient, which: str, count: int, min_mixing_n: int | None = None, log_progress: bool = False) -> list[int]:
  idx_list: list[int] = []
  if min_mixing_n is not None:
    last_log_time = 0.
    wait_start_time = time.perf_counter()
    scard = coord.count(which)
    wait_start_scard = scard
    while scard < min_mixing_n:
      now = time.perf_counter()
      if log_progress and now - last_log_time >= LOG_INTERVAL_S:
        elapsed_s, eta_s = get_elapsed_and_eta(now, wait_start_time, scard, wait_start_scard, min_mixing_n)
        logger.info(f"waiting for {which} - {scard} / {min_mixing_n} ({elapsed_s:.0f}s/{eta_s:.0f}s)")
        last_log_time = now
      time.sleep(0.1)
      scard = coord.count(which)
    if log_progress:
      logger.info(f"{which} reached {min_mixing_n}")
  while True:
    idx_list.extend(coord.pop(which, count - len(idx_list)))
    if len(idx_list) >= count:
      break
    time.sleep(0.1)
  return idx_list


def wait_for_shuffle_buffer_attach_count(coord: CoordinatorClient, queue_name: str, expected_count: int) -> None:
  last_log_time = 0.
  while (attached_count := coord.attached_count()) < expected_count:
    if time.perf_counter() - last_log_time >= LOG_INTERVAL_S:
      logger.info(f"waiting for {queue_name} attached processes - {attached_count} / {expected_count}")
      last_log_time = time.perf_counter()
    time.sleep(0.1)


def create_shared_shuffle_buffer_attachment(first_samples: Buffer, shuffle_size: int, input_bs: int, input_bs_key: tuple[int, str], queue_name: str, dummy_bs: int, print_shapes: bool = True) -> ShuffleBufferAttachment:
  metadata: ShuffleBufferMetadata = dict(queue_name=queue_name, shuffle_size=shuffle_size, input_bs=input_bs, input_bs_key=input_bs_key, fields=[])
  shuffle_buffer = []

  for i in range(len(first_samples)):
    b = {}
    for k,v in first_samples[i].items():
      dtype = numpy_type_to_torch(v.dtype)
      shape = tuple([shuffle_size]+list(v.shape[1:]))
      tensor = torch.empty(shape, dtype=dtype).share_memory_()
      if print_shapes:
        logger.info(f"allocating shared shape {list(shape)} for {k} with type {dtype}")
      b[k] = tensor
      metadata['fields'].append(dict(i=i, k=k, shape=shape, dtype=str(dtype).removeprefix('torch.'), storage_offset=tensor.storage_offset(), stride=tensor.stride()))
    shuffle_buffer.append(b)

  dummy_batch = []
  idxs = (np.arange(dummy_bs) % input_bs).tolist()
  for i in range(len(first_samples)):
    b = {}
    for k,v in first_samples[i].items():
      dtype = numpy_type_to_torch(v.dtype)
      shape = tuple([dummy_bs]+list(v.shape[1:]))
      tensor = torch.empty(shape, dtype=dtype).share_memory_()
      tmp = torch.as_tensor(v[idxs])
      if tmp.device != tensor.device or tmp.dtype != tensor.dtype:
        tmp = tmp.to(device=tensor.device, dtype=tensor.dtype)
      tensor[:] = tmp
      b[k] = tensor
    dummy_batch.append(b)
  dummy_batch[0][INDEX_KEY] = torch.full((dummy_bs,), -1, dtype=torch.int64).share_memory_()

  return ShuffleBufferAttachment(metadata=metadata, shuffle_buffer=shuffle_buffer, dummy_batch=dummy_batch)


def start_coordinator(queue_name: str, attachment: ShuffleBufferAttachment, empty_indices: list[int]) -> None:
  CoordinatorServer(queue_name, attachment, empty_indices).start()


def attach_to_shared_shuffle_buffer(queue_name: str, count_attach: bool = True, check_children: Any | None = None) -> ShuffleBufferAttachment:
  initialize_shuffle_buffer_tensor_ipc()
  coord = CoordinatorClient(queue_name)
  last_log_time = 0.

  while True:
    try:
      return cast(ShuffleBufferAttachment, coord.attach(count_attach=count_attach))
    except COORDINATOR_CONNECT_ERRORS as e:
      if check_children is not None:
        check_children()
      if time.perf_counter() - last_log_time >= LOG_INTERVAL_S:
        logger.info(f"waiting for gigashuffle coordinator for {queue_name}: {e}")
        last_log_time = time.perf_counter()
      time.sleep(0.05)


def initialize_writer(dset: Dataset, config: DataloaderConfig, proc_idx: int, queue_name: str) -> tuple[CoordinatorClient, Any, Buffer, ShuffleBufferMetadata]:
  init_logger()
  initialize_shuffle_buffer_tensor_ipc()
  setproctitle(f'gigashuffle writer {queue_name} local_rank={config.local_rank} proc={proc_idx}')
  os.system(f'renice -n 3 -p {os.getpid()} > /dev/null')

  shuffle_size = config.shuffle_size
  global_proc_idx = config.global_rank * config.num_writers + proc_idx
  total_procs = config.global_world_size * config.num_writers
  local_proc_idx = config.local_rank * config.num_writers + proc_idx
  random.seed(global_proc_idx)
  torch.manual_seed(global_proc_idx)
  np.random.seed(global_proc_idx)
  set_worker_info(dset, worker_id=global_proc_idx, num_workers=total_procs, seed=global_proc_idx)

  coord = CoordinatorClient(queue_name)
  dset_iter = iter(dset) if hasattr(dset, '__iter__') else dset
  if local_proc_idx == 0:
    input_samples, input_bs, input_bs_key = fetch_initial_sample(dset_iter, config)
    attachment = create_shared_shuffle_buffer_attachment(input_samples, shuffle_size, input_bs, input_bs_key, queue_name, config.bs)
    initial_idx_list = list(range(input_bs))
    start_coordinator(queue_name, attachment, list(range(input_bs, shuffle_size)))
    expected_attach_count = config.local_world_size * (config.num_writers + config.num_readers) - 1
    wait_for_shuffle_buffer_attach_count(coord, queue_name, expected_attach_count)
    for i in range(len(attachment.shuffle_buffer)):
      for k in attachment.shuffle_buffer[i].keys():
        tmp = torch.as_tensor(input_samples[i][k])
        if tmp.device != attachment.shuffle_buffer[i][k].device or tmp.dtype != attachment.shuffle_buffer[i][k].dtype:
          tmp = tmp.to(device=attachment.shuffle_buffer[i][k].device, dtype=attachment.shuffle_buffer[i][k].dtype)
        attachment.shuffle_buffer[i][k][initial_idx_list] = tmp
    coord.push('full', initial_idx_list)
  else:
    attachment = attach_to_shared_shuffle_buffer(queue_name)

  logger.info(f"writer {config.global_rank}-{proc_idx} initialized with input_bs {attachment.metadata['input_bs']} output_bs {config.bs}")
  return coord, dset_iter, attachment.shuffle_buffer, attachment.metadata


def write_samples_to_buffer(shuffle_buffer: Buffer, samples: Buffer, idx_list: list[int], local_input_bs: int) -> None:
  for i in range(len(shuffle_buffer)):
    for k in shuffle_buffer[i].keys():
      tmp = torch.as_tensor(samples[i][k])[:local_input_bs]
      if tmp.device != shuffle_buffer[i][k].device or tmp.dtype != shuffle_buffer[i][k].dtype:
        tmp = tmp.to(device=shuffle_buffer[i][k].device, dtype=shuffle_buffer[i][k].dtype)
      shuffle_buffer[i][k][idx_list] = tmp


def streaming_writer(dset: Dataset, config: DataloaderConfig, proc_idx: int, queue_name: str, parent_pid: int) -> None:
  set_parent_death_signal(parent_pid)
  coord, dset_iter, shuffle_buffer, metadata = initialize_writer(dset, config, proc_idx, queue_name)
  rank_id = RANK_ID_FORMAT.format(global_rank=config.global_rank)
  while True:
    training_context = coord.get_context(rank_id)
    if training_context is not None:
      dset.context = pickle.loads(training_context)
    samples, local_input_bs, _ = get_samples(dset_iter, metadata['input_bs_key'], max_retries=config.writer_max_retries)
    max_input_bs = (config.shuffle_size - config.bs) // (config.local_world_size * config.num_writers)
    if local_input_bs > max_input_bs:
      local_input_bs = max_input_bs
      print_small_shuffle_warning()
    idx_list = fetch_rand_from_queue(coord, 'empty', local_input_bs)
    write_samples_to_buffer(shuffle_buffer, samples, idx_list, local_input_bs)
    coord.push('full', idx_list)


def exit_or_keep_coordinator_alive(owns_coordinator: bool) -> None:
  if owns_coordinator:
    # Keep the coordinator available for late dummy-batch requests.
    while True:
      time.sleep(3600)
  raise SystemExit(FILL_ONCE_WRITER_DONE_EXITCODE)


def fill_once_writer(dset: Dataset, config: DataloaderConfig, proc_idx: int, queue_name: str, parent_pid: int) -> None:
  set_parent_death_signal(parent_pid)
  coord, dset_iter, shuffle_buffer, metadata = initialize_writer(dset, config, proc_idx, queue_name)
  owns_coordinator = config.local_rank * config.num_writers + proc_idx == 0
  rank_id = RANK_ID_FORMAT.format(global_rank=config.global_rank)
  while True:
    training_context = coord.get_context(rank_id)
    if training_context is not None:
      dset.context = pickle.loads(training_context)
    empty_n = coord.count('empty')
    if empty_n == 0:
      exit_or_keep_coordinator_alive(owns_coordinator)
    samples, local_input_bs, _ = get_samples(dset_iter, metadata['input_bs_key'], max_retries=config.writer_max_retries)
    local_input_bs = min(local_input_bs, empty_n)
    idx_list = coord.pop('empty', local_input_bs)
    if not idx_list:
      exit_or_keep_coordinator_alive(owns_coordinator)
    local_input_bs = len(idx_list)
    write_samples_to_buffer(shuffle_buffer, samples, idx_list, local_input_bs)
    coord.push('full', idx_list)


def initialize_reader(config: DataloaderConfig, proc_idx: int, queue_name: str) -> tuple[CoordinatorClient, Buffer, Buffer]:
  init_logger()
  setproctitle(f'gigashuffle reader {queue_name} local_rank={config.local_rank} proc={proc_idx}')
  coord = CoordinatorClient(queue_name)
  attachment = attach_to_shared_shuffle_buffer(queue_name)
  reader_buffer: Buffer = [{} for _ in range(max(t['i'] for t in attachment.metadata['fields']) + 1)]
  for t in attachment.metadata['fields']:
    dtype = getattr(torch, t['dtype'])
    shape = tuple([config.bs] + list(t['shape'][1:]))
    reader_buffer[t['i']][t['k']] = torch.empty(shape, dtype=dtype).share_memory_()
  reader_buffer[0][INDEX_KEY] = torch.empty(config.bs, dtype=torch.int64).share_memory_()
  return coord, attachment.shuffle_buffer, reader_buffer


def copy_to_reader_buffer(reader_buffer: Buffer, shuffle_buffer: Buffer, idx_list: list[int]) -> None:
  for buffer_idx in range(len(shuffle_buffer)):
    for k in shuffle_buffer[buffer_idx].keys():
      reader_buffer[buffer_idx][k][:] = shuffle_buffer[buffer_idx][k][idx_list]
  reader_buffer[0][INDEX_KEY].copy_(torch.as_tensor(idx_list))


def send_reader_buffer(ready_q: SimpleQueue[tuple[Buffer, int]], ready_e: Event, reader_buffer: Buffer, proc_idx: int) -> None:
  ready_q.put((reader_buffer, proc_idx))
  while not ready_e.is_set():
    ready_e.wait()
  ready_e.clear()


def streaming_reader(config: DataloaderConfig, ready_q: SimpleQueue[tuple[Buffer, int]], ready_e: Event, request_batch_q: SimpleQueue[int] | None, proc_idx: int, queue_name: str, parent_pid: int):
  assert request_batch_q is None
  set_parent_death_signal(parent_pid)
  coord, shuffle_buffer, reader_buffer = initialize_reader(config, proc_idx, queue_name)
  min_mixing_n = int(config.min_mixing * config.shuffle_size)
  return_queue = 'empty' if config.evict_on_read else 'full'

  for batch_idx in count():
    idx_list = fetch_rand_from_queue(coord, 'full', config.bs, min_mixing_n=min_mixing_n, log_progress=batch_idx == 0 and config.local_rank == 0 and proc_idx == 0)
    copy_to_reader_buffer(reader_buffer, shuffle_buffer, idx_list)
    coord.push(return_queue, idx_list)
    send_reader_buffer(ready_q, ready_e, reader_buffer, proc_idx)


def fill_once_reader(config: DataloaderConfig, ready_q: SimpleQueue[tuple[Buffer, int]], ready_e: Event, request_batch_q: SimpleQueue[int] | None, proc_idx: int, queue_name: str, parent_pid: int):
  assert request_batch_q is not None
  set_parent_death_signal(parent_pid)
  coord, shuffle_buffer, reader_buffer = initialize_reader(config, proc_idx, queue_name)

  last_log_time = 0.
  wait_start_time = time.perf_counter()
  scard = coord.count('full')
  wait_start_scard = scard
  while scard < config.shuffle_size:
    now = time.perf_counter()
    if config.local_rank == 0 and now - last_log_time >= LOG_INTERVAL_S:
      elapsed_s, eta_s = get_elapsed_and_eta(now, wait_start_time, scard, wait_start_scard, config.shuffle_size)
      logger.info(f"waiting for {queue_name}-full - {scard} / {config.shuffle_size} ({elapsed_s:.0f}s/{eta_s:.0f}s)")
      last_log_time = now
    time.sleep(0.1)
    scard = coord.count('full')
  if config.local_rank == 0:
    logger.info(f"{queue_name}-full reached {config.shuffle_size}")

  while True:
    batch_idx = request_batch_q.get()
    start_idx = (batch_idx * config.local_world_size + config.local_rank) * config.bs
    idx_list = list(range(start_idx, start_idx + config.bs))
    copy_to_reader_buffer(reader_buffer, shuffle_buffer, idx_list)
    send_reader_buffer(ready_q, ready_e, reader_buffer, proc_idx)


class MultiprocessShuffledDataloader(IterableDataset):
  def __init__(self, dset: Dataset, config: DataloaderConfig) -> None:
    init_logger()
    self.dset = dset
    self.config = config
    assert config.num_writers > 0, "gigashuffle requires num_writers > 0"
    assert config.queue_name, "MultiprocessShuffledDataloader requires config.queue_name"
    assert not (config.fill_once and not config.evict_on_read), "evict_on_read=False is not supported with fill_once"
    if config.fill_once:
      assert config.num_readers == 1, "fill_once requires num_readers == 1"
      assert config.min_mixing == 1, "fill_once requires min_mixing == 1"
      assert config.shuffle_size % (config.bs * config.local_world_size) == 0, "fill_once requires shuffle_size to be divisible by bs * local_world_size"
    self.max_iters = config.shuffle_size // (config.bs * config.local_world_size) if config.fill_once else None
    self.queue_name = f'gigashuffle-{config.queue_name}'
    initialize_shuffle_buffer_tensor_ipc()
    self._dummy_batch: Buffer | None = None
    self._rank_id = RANK_ID_FORMAT.format(global_rank=config.global_rank)
    self._shutdown = False
    self._coord = CoordinatorClient(self.queue_name)

    ctx = mp.get_context('spawn')
    self.ready_q: SimpleQueue[tuple[Buffer, int]] = ctx.SimpleQueue()
    self.ready_e = [ctx.Event() for _ in range(self.config.num_readers)]
    self.request_batch_q: SimpleQueue[int] | None = ctx.SimpleQueue() if config.fill_once else None
    self.children = []
    self.check_child_time = 0.

    reader_fn = fill_once_reader if config.fill_once else streaming_reader
    writer_fn = fill_once_writer if config.fill_once else streaming_writer
    parent_pid = os.getpid()
    for i in range(self.config.num_readers):
      args = (config, self.ready_q, self.ready_e[i], self.request_batch_q, i, self.queue_name, parent_pid)
      self.children.append(ctx.Process(target=reader_fn, args=args, daemon=True))
    for i in range(self.config.num_writers):
      self.children.append(ctx.Process(target=writer_fn, args=(dset, config, i, self.queue_name, parent_pid), daemon=True))

    for i, p in enumerate(self.children):
      p.start()

  def _coord_call(self, fn, *args):
    while True:
      try:
        return fn(*args)
      except COORDINATOR_CONNECT_ERRORS:
        if time.perf_counter() - self.check_child_time > 1.0:
          self.check_children()
        time.sleep(0.05)

  def attach_training_context(self, context: Any) -> None:
    self._coord_call(self._coord.set_context, self._rank_id, pickle.dumps(context))

  def state_dict(self) -> dict[str, Any]:
    state = {}
    if hasattr(self.dset, 'state_dict'):
      state['dataset'] = self.dset.state_dict()
    return {self._rank_id: pickle.dumps(state), 'world_size': self.config.global_world_size}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    if not state_dict:
      return
    if self._rank_id not in state_dict:
      logger.warning(f"MultiprocessShuffledDataloader state is empty for global rank {self.config.global_rank}, expected key {self._rank_id}")
      return
    assert self.config.global_world_size == state_dict['world_size'], "global_world_size is inconsistent before and after checkpoint, dataloader resharding is not supported yet."
    state = pickle.loads(state_dict[self._rank_id])
    if 'dataset' in state and hasattr(self.dset, 'load_state_dict'):
      self.dset.load_state_dict(state['dataset'])

  def get_dummy_batch(self) -> Buffer:
    if self._dummy_batch is None:
      attachment = attach_to_shared_shuffle_buffer(self.queue_name, count_attach=False, check_children=self.check_children)
      self._dummy_batch = attachment.dummy_batch
    return self._dummy_batch

  def stats(self) -> ShuffleBufferStats:
    stats = self._coord_call(self._coord.stats)
    return ShuffleBufferStats(full=stats['full'], empty=stats['empty'], in_flight=stats['in_flight'])

  def evict(self, indices: list[int]) -> int:
    assert not self.config.evict_on_read, "evict() requires evict_on_read=False"
    if not (indices := [int(i) for i in indices]):
      return 0
    return self._coord_call(self._coord.evict, indices)

  def check_children(self) -> None:
    for i, p in enumerate(self.children):
      if not p.is_alive():
        if self.config.fill_once and i >= self.config.num_readers and p.exitcode == FILL_ONCE_WRITER_DONE_EXITCODE:
          continue
        raise RuntimeError(f"MultiprocessShuffledDataloader child {p.name} (pid={p.pid}) died (exitcode={p.exitcode}). Aborting.")
    self.check_child_time = time.perf_counter()

  def _shutdown_workers(self) -> None:
    if globals().get('_python_exit_status') is not False:
      return
    if self._shutdown:
      return
    self._shutdown = True
    children = getattr(self, 'children', [])
    try:
      for p in children:
        if p.is_alive():
          p.terminate()
      join_deadline = time.perf_counter() + CLOSE_JOIN_TIMEOUT_S
      for p in children:
        p.join(timeout=max(0, join_deadline - time.perf_counter()))
    finally:
      for p in children:
        if p.is_alive():
          p.kill()
      kill_deadline = time.perf_counter() + CLOSE_JOIN_TIMEOUT_S
      for p in children:
        p.join(timeout=max(0, kill_deadline - time.perf_counter()))
      if ready_q := getattr(self, 'ready_q', None):
        ready_q.close()
      if request_batch_q := getattr(self, 'request_batch_q', None):
        request_batch_q.close()

  def __del__(self) -> None:
    try:
      self._shutdown_workers()
    except Exception:
      pass

  def __iter__(self) -> Iterator[Buffer]:
    yielded = 0
    if self.config.fill_once:
      max_iters = cast(int, self.max_iters)
      assert self.request_batch_q is not None
      self.ready_e[0].set()
      self.request_batch_q.put(0)

    while True:
      if time.perf_counter() - self.check_child_time > 1.0:
        self.check_children()

      if not self.ready_q.empty():
        buf, idx = self.ready_q.get()
      else:
        time.sleep(0.001)
        continue

      if self.config.fill_once:
        yield buf
        self.ready_e[idx].set()
        yielded += 1
        if yielded >= max_iters:
          return
        self.request_batch_q.put(yielded)
      else:
        try:
          yield buf
        finally:
          self.ready_e[idx].set()
