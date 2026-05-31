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
from itertools import batched, count
from multiprocessing import resource_tracker
from multiprocessing.synchronize import Event
from multiprocessing.queues import SimpleQueue
from typing import Any, Iterator, cast

import torch
import numpy as np
import torch.multiprocessing as mp
from redis import StrictRedis
from setproctitle import setproctitle
from torch.utils.data import Dataset, IterableDataset
from gigashuffle.worker_info import set_worker_info
from gigashuffle.config import DataloaderConfig


Buffer = list[dict[str, torch.Tensor]]
CHUNK_SIZE = 1024*64
LOG_INTERVAL_S = 5.0
PR_SET_PDEATHSIG = 1
FILL_ONCE_WRITER_DONE_EXITCODE = 81
CLOSE_JOIN_TIMEOUT_S = 0.2
ShuffleBufferMetadata = dict[str, Any]
logger = logging.getLogger(__name__)
already_warned = False
_python_exit_status = False


def _set_python_exit_status() -> None:
  global _python_exit_status
  _python_exit_status = True


atexit.register(_set_python_exit_status)

# NOTE: For high-throughput tasks, calling torch.set_num_threads(1) seems to significantly reduce CPU usage in the gigashuffle writers.
# This SO thread about a similar issue notes that it "was caused by a bad interaction of OpenMP and multiprocessing".
# https://stackoverflow.com/questions/65057388/pytorch-multiprocessing-with-shared-memory-causes-matmul-to-be-30x-slower-with
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


def set_process_title(kind: str, config: DataloaderConfig, proc_idx: int, queue_name: str) -> None:
  setproctitle(f'gigashuffle {kind} {queue_name} local_rank={config.local_rank} proc={proc_idx}')


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
  else:
    raise Exception("unsupported numpy type %r" % x)


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

  logger.info("each element uses %d bytes, total buffer size is %.3fgb", memory_size, shuffle_size*memory_size/1e9)
  if shuffle_size < config.bs * config.local_world_size:
    N = config.local_world_size * config.num_readers
    raise RuntimeError(f"Shuffle buffer must be large enough to accommodate at least N batches, but buffer size = {shuffle_size}, batch size = {config.bs}, N = {N}")
  if not config.fill_once and min_mixing_n >= shuffle_size - 2*input_bs:
    raise RuntimeError(f"To avoid deadlock, min_mixing_n ({min_mixing_n}) must be less than {shuffle_size - 2*input_bs}")

  return input_samples, input_bs, input_bs_key


def initialize_redis_queue(r: StrictRedis, queue_name: str, shuffle_size: int) -> None:
  logger.info("setting up %s on redis version %s", queue_name, r.execute_command('INFO')['redis_version'])
  r.delete(f"{queue_name}-shared-buffer-meta")
  r.delete(f"{queue_name}-shared-buffer-attached")
  r.delete(f"{queue_name}-empty")
  r.delete(f"{queue_name}-full")
  for idxs in batched(range(0, shuffle_size), n=CHUNK_SIZE):
    r.sadd(f'{queue_name}-empty', *idxs)


def fetch_rand_from_queue(r: StrictRedis, queue_name: str, count: int, min_mixing_n: int | None = None, log_progress: bool = False) -> list[int]:
  idx_list: list[int] = []
  if min_mixing_n is not None:
    last_log_time = 0.
    while (scard := cast(int, r.scard(queue_name))) < min_mixing_n:
      if log_progress and time.perf_counter() - last_log_time >= LOG_INTERVAL_S:
        logger.info("waiting for %s - %d / %d", queue_name, scard, min_mixing_n)
        last_log_time = time.perf_counter()
      time.sleep(0.1)
    if log_progress:
      logger.info("%s reached min_mixing_n=%d", queue_name, min_mixing_n)
  while True:
    idx_list.extend(int(x) for x in cast(list[bytes], r.spop(queue_name, count - len(idx_list))))
    if len(idx_list) >= count:
      break
    time.sleep(0.1)
  return idx_list


def wait_for_shuffle_buffer_attachments(r: StrictRedis, queue_name: str, expected_count: int) -> None:
  attached_key = f'{queue_name}-shared-buffer-attached'
  last_log_time = 0.
  while (attached_count := int(cast(bytes | int | None, r.get(attached_key)) or 0)) < expected_count:
    if time.perf_counter() - last_log_time >= LOG_INTERVAL_S:
      logger.info("waiting for %s - %d / %d", attached_key, attached_count, expected_count)
      last_log_time = time.perf_counter()
    time.sleep(0.1)


def prevent_resource_tracker_fd_inheritance() -> None:
  # torch_shm_manager otherwise inherits this pipe and can block Python shutdown.
  try:
    os.set_inheritable(resource_tracker.getfd(), False)
  except Exception:
    logger.debug("failed to mark resource tracker fd close-on-exec", exc_info=True)


def get_shared_storage_metadata(tensor: torch.Tensor) -> dict[str, Any]:
  prevent_resource_tracker_fd_inheritance()
  storage = tensor.untyped_storage()
  manager, handle, size = storage._share_filename_cpu_()
  return dict(manager=manager, handle=handle, size=size, storage_offset=tensor.storage_offset(), stride=tensor.stride())


def create_shuffle_buffer(first_samples: Buffer, shuffle_size: int, input_bs: int, input_bs_key: tuple[int, str], queue_name: str, dummy_bs: int, print_shapes: bool = True) -> tuple[Buffer, Buffer, ShuffleBufferMetadata]:
  shuffle_buffer_metadata: ShuffleBufferMetadata = dict(queue_name=queue_name, shuffle_size=shuffle_size, input_bs=input_bs, input_bs_key=input_bs_key, fields=[], dummy_fields=[])
  shuffle_buffer = []

  for i in range(len(first_samples)):
    b = {}
    for k,v in first_samples[i].items():
      dtype = numpy_type_to_torch(v.dtype)
      shape = tuple([shuffle_size]+list(v.shape[1:]))
      tensor = torch.empty(shape, dtype=dtype).share_memory_()
      if print_shapes:
        logger.info("allocating shared shape %s for %s with type %s", list(shape), k, dtype)
      b[k] = tensor
      shuffle_buffer_metadata['fields'].append(dict(i=i, k=k, shape=shape, dtype=str(dtype).removeprefix('torch.'), **get_shared_storage_metadata(tensor)))
    shuffle_buffer.append(b)

  dummy_buffer = []
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
      shuffle_buffer_metadata['dummy_fields'].append(dict(i=i, k=k, shape=shape, dtype=str(dtype).removeprefix('torch.'), **get_shared_storage_metadata(tensor)))
    dummy_buffer.append(b)

  return shuffle_buffer, dummy_buffer, shuffle_buffer_metadata


def attach_shared_tensor(t: dict[str, Any]) -> torch.Tensor:
  dtype = getattr(torch, t['dtype'])
  prevent_resource_tracker_fd_inheritance()
  storage = torch.UntypedStorage._new_shared_filename_cpu(t['manager'], t['handle'], t['size'])
  return torch.tensor([], dtype=dtype).set_(storage, t['storage_offset'], tuple(t['shape']), tuple(t['stride']))


def attach_buffer_fields(fields: list[dict[str, Any]], bs: int | None = None, shared: bool = False) -> Buffer:
  shuffle_buffer: Buffer = [{} for _ in range(max(t['i'] for t in fields) + 1)]
  for t in fields:
    dtype = getattr(torch, t['dtype'])
    shape = tuple(([bs] if bs is not None else [t['shape'][0]]) + list(t['shape'][1:]))
    if bs is None:
      shuffle_buffer[t['i']][t['k']] = attach_shared_tensor(t)
    else:
      tensor = torch.empty(shape, dtype=dtype)
      shuffle_buffer[t['i']][t['k']] = tensor.share_memory_() if shared else tensor
  return shuffle_buffer


def wait_for_shuffle_buffer_metadata(r: StrictRedis, queue_name: str) -> ShuffleBufferMetadata:
  while True:
    raw = cast(bytes | None, r.get(f'{queue_name}-shared-buffer-meta'))
    if raw is not None:
      return cast(ShuffleBufferMetadata, pickle.loads(raw))
    time.sleep(0.1)


def initialize_writer(dset: Dataset, config: DataloaderConfig, proc_idx: int, queue_name: str) -> tuple[StrictRedis, Any, Buffer, Buffer | None, ShuffleBufferMetadata]:
  init_logger()
  setproctitle('gigashuffle writer %d %d' % (config.local_rank, proc_idx))
  os.system('renice -n 3 -p %d > /dev/null' % os.getpid())

  shuffle_size = config.shuffle_size
  global_proc_idx = config.global_rank * config.num_writers + proc_idx
  total_procs = config.global_world_size * config.num_writers
  local_proc_idx = config.local_rank * config.num_writers + proc_idx
  random.seed(global_proc_idx)
  torch.manual_seed(global_proc_idx)
  np.random.seed(global_proc_idx)
  set_worker_info(dset, worker_id=global_proc_idx, num_workers=total_procs, seed=global_proc_idx)

  r = StrictRedis(host=config.redis_host, port=config.redis_port, db=config.redis_db)
  dset_iter = iter(dset) if hasattr(dset, '__iter__') else dset
  if local_proc_idx == 0:
    initialize_redis_queue(r, queue_name, shuffle_size)
    input_samples, input_bs, input_bs_key = fetch_initial_sample(dset_iter, config)
    shuffle_buffer, dummy_buffer, shuffle_buffer_metadata = create_shuffle_buffer(input_samples, shuffle_size, input_bs, input_bs_key, queue_name, config.bs)
    initial_idx_list = list(range(input_bs))
    r.srem(f'{queue_name}-empty', *initial_idx_list)
    r.set(f'{queue_name}-shared-buffer-meta', pickle.dumps(shuffle_buffer_metadata))
    expected_attach_count = config.local_world_size * (config.num_writers + config.num_readers) - 1
    wait_for_shuffle_buffer_attachments(r, queue_name, expected_attach_count)
    for i in range(len(shuffle_buffer)):
      for k in shuffle_buffer[i].keys():
        tmp = torch.as_tensor(input_samples[i][k])
        if tmp.device != shuffle_buffer[i][k].device or tmp.dtype != shuffle_buffer[i][k].dtype:
          tmp = tmp.to(device=shuffle_buffer[i][k].device, dtype=shuffle_buffer[i][k].dtype)
        shuffle_buffer[i][k][initial_idx_list] = tmp
    r.sadd(f'{queue_name}-full', *initial_idx_list)
  else:
    shuffle_buffer_metadata = wait_for_shuffle_buffer_metadata(r, queue_name)
    shuffle_buffer = attach_buffer_fields(shuffle_buffer_metadata['fields'])
    dummy_buffer = None
    r.incr(f'{queue_name}-shared-buffer-attached')

  logger.info("writer %d-%d initialized with input_bs %d output_bs %d", config.global_rank, proc_idx, shuffle_buffer_metadata['input_bs'], config.bs)
  return r, dset_iter, shuffle_buffer, dummy_buffer, shuffle_buffer_metadata


def write_samples_to_buffer(shuffle_buffer: Buffer, samples: Buffer, idx_list: list[int], local_input_bs: int) -> None:
  for i in range(len(shuffle_buffer)):
    for k in shuffle_buffer[i].keys():
      tmp = torch.as_tensor(samples[i][k])[:local_input_bs]
      if tmp.device != shuffle_buffer[i][k].device or tmp.dtype != shuffle_buffer[i][k].dtype:
        tmp = tmp.to(device=shuffle_buffer[i][k].device, dtype=shuffle_buffer[i][k].dtype)
      shuffle_buffer[i][k][idx_list] = tmp


def streaming_writer(dset: Dataset, config: DataloaderConfig, proc_idx: int, queue_name: str, parent_pid: int) -> None:
  set_parent_death_signal(parent_pid)
  r, dset_iter, shuffle_buffer, _dummy_buffer, shuffle_buffer_metadata = initialize_writer(dset, config, proc_idx, queue_name)
  empty_key = f'{queue_name}-empty'
  while True:
    samples, local_input_bs, _ = get_samples(dset_iter, shuffle_buffer_metadata['input_bs_key'], max_retries=config.writer_max_retries)
    max_input_bs = (config.shuffle_size - config.bs) // (config.local_world_size * config.num_writers)
    if local_input_bs > max_input_bs:
      local_input_bs = max_input_bs
      print_small_shuffle_warning()
    idx_list = fetch_rand_from_queue(r, empty_key, local_input_bs)
    write_samples_to_buffer(shuffle_buffer, samples, idx_list, local_input_bs)
    r.sadd(f'{queue_name}-full', *idx_list)


def fill_once_writer(dset: Dataset, config: DataloaderConfig, proc_idx: int, queue_name: str, parent_pid: int) -> None:
  set_parent_death_signal(parent_pid)
  r, dset_iter, shuffle_buffer, _dummy_buffer, shuffle_buffer_metadata = initialize_writer(dset, config, proc_idx, queue_name)
  empty_key = f'{queue_name}-empty'
  while True:
    empty_n = cast(int, r.scard(empty_key))
    if empty_n == 0:
      raise SystemExit(FILL_ONCE_WRITER_DONE_EXITCODE)
    samples, local_input_bs, _ = get_samples(dset_iter, shuffle_buffer_metadata['input_bs_key'], max_retries=config.writer_max_retries)
    local_input_bs = min(local_input_bs, empty_n)
    idx_list = [int(x) for x in cast(list[bytes], r.spop(empty_key, local_input_bs))]
    if not idx_list:
      raise SystemExit(FILL_ONCE_WRITER_DONE_EXITCODE)
    local_input_bs = len(idx_list)
    write_samples_to_buffer(shuffle_buffer, samples, idx_list, local_input_bs)
    r.sadd(f'{queue_name}-full', *idx_list)


def initialize_reader(config: DataloaderConfig, proc_idx: int, queue_name: str) -> tuple[StrictRedis, Buffer, Buffer]:
  init_logger()
  set_process_title('reader', config, proc_idx, queue_name)
  r = StrictRedis(host=config.redis_host, port=config.redis_port, db=config.redis_db)
  shuffle_buffer_metadata = wait_for_shuffle_buffer_metadata(r, queue_name)
  shuffle_buffer = attach_buffer_fields(shuffle_buffer_metadata['fields'])
  r.incr(f'{queue_name}-shared-buffer-attached')
  reader_buffer = attach_buffer_fields(shuffle_buffer_metadata['fields'], bs=config.bs, shared=True)
  return r, shuffle_buffer, reader_buffer


def copy_to_reader_buffer(reader_buffer: Buffer, shuffle_buffer: Buffer, idx_list: list[int]) -> None:
  for buffer_idx in range(len(shuffle_buffer)):
    for k in shuffle_buffer[buffer_idx].keys():
      reader_buffer[buffer_idx][k][:] = shuffle_buffer[buffer_idx][k][idx_list]


def send_reader_buffer(ready_q: SimpleQueue[tuple[Buffer, int]], ready_e: Event, reader_buffer: Buffer, proc_idx: int) -> None:
  ready_q.put((reader_buffer, proc_idx))
  while not ready_e.is_set():
    ready_e.wait()
  ready_e.clear()


def streaming_reader(config: DataloaderConfig, ready_q: SimpleQueue[tuple[Buffer, int]], ready_e: Event, proc_idx: int, queue_name: str, parent_pid: int):
  set_parent_death_signal(parent_pid)
  r, shuffle_buffer, reader_buffer = initialize_reader(config, proc_idx, queue_name)
  min_mixing_n = int(config.min_mixing * config.shuffle_size)

  for batch_idx in count():
    idx_list = fetch_rand_from_queue(r, f'{queue_name}-full', config.bs, min_mixing_n=min_mixing_n, log_progress=batch_idx == 0 and config.local_rank == 0 and proc_idx == 0)
    copy_to_reader_buffer(reader_buffer, shuffle_buffer, idx_list)
    r.sadd(f'{queue_name}-empty', *idx_list)
    send_reader_buffer(ready_q, ready_e, reader_buffer, proc_idx)


def fill_once_reader(config: DataloaderConfig, ready_q: SimpleQueue[tuple[Buffer, int]], ready_e: Event, proc_idx: int, queue_name: str, parent_pid: int):
  set_parent_death_signal(parent_pid)
  r, shuffle_buffer, reader_buffer = initialize_reader(config, proc_idx, queue_name)

  last_log_time = 0.
  full_key = f'{queue_name}-full'
  while (scard := cast(int, r.scard(full_key))) < config.shuffle_size:
    if config.local_rank == 0 and time.perf_counter() - last_log_time >= LOG_INTERVAL_S:
      logger.info("waiting for %s - %d / %d", full_key, scard, config.shuffle_size)
      last_log_time = time.perf_counter()
    time.sleep(0.1)
  if config.local_rank == 0:
    logger.info("%s reached min_mixing_n=%d", full_key, config.shuffle_size)

  for batch_idx in count():
    start_idx = (batch_idx * config.local_world_size + config.local_rank) * config.bs % config.shuffle_size
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
    if config.fill_once:
      assert config.num_readers == 1, "fill_once requires num_readers == 1"
      assert config.min_mixing == 1, "fill_once requires min_mixing == 1"
      assert config.shuffle_size % (config.bs * config.local_world_size) == 0, "fill_once requires shuffle_size to be divisible by bs * local_world_size"
    self.max_iters = config.shuffle_size // (config.bs * config.local_world_size) if config.fill_once else None
    self.queue_name = f'gigashuffle-{config.queue_name}'
    self._shuffle_buffer_metadata: ShuffleBufferMetadata | None = None
    self._dummy_batch: Buffer | None = None
    self._rank_id = f'global_rank_{config.global_rank}'
    self._shutdown = False

    self._r = StrictRedis(host=config.redis_host, port=config.redis_port, db=config.redis_db)
    try:
      assert self._r.ping()
    except Exception as e:
      raise AssertionError(f"Redis is not reachable at {config.redis_host}:{config.redis_port}/{config.redis_db}") from e

    ctx = mp.get_context('spawn')
    self.ready_q: SimpleQueue[tuple[Buffer, int]] = ctx.SimpleQueue()
    self.ready_e = [ctx.Event() for _ in range(self.config.num_readers)]
    self.children = []
    self.check_child_time = 0.

    reader_fn = fill_once_reader if config.fill_once else streaming_reader
    writer_fn = fill_once_writer if config.fill_once else streaming_writer
    parent_pid = os.getpid()
    for i in range(self.config.num_readers):
      self.children.append(ctx.Process(target=reader_fn, args=(config, self.ready_q, self.ready_e[i], i, self.queue_name, parent_pid), daemon=True))
    for i in range(self.config.num_writers):
      self.children.append(ctx.Process(target=writer_fn, args=(dset, config, i, self.queue_name, parent_pid), daemon=True))

    for i, p in enumerate(self.children):
      p.start()

  @property
  def shuffle_buffer_metadata(self) -> ShuffleBufferMetadata:
    if self._shuffle_buffer_metadata is None:
      self._shuffle_buffer_metadata = wait_for_shuffle_buffer_metadata(self._r, self.queue_name)
    return self._shuffle_buffer_metadata

  def state_dict(self) -> dict[str, Any]:
    state = {}
    if hasattr(self.dset, 'state_dict'):
      state['dataset'] = self.dset.state_dict()
    return {self._rank_id: pickle.dumps(state), 'world_size': self.config.global_world_size}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    if not state_dict:
      return
    if self._rank_id not in state_dict:
      logger.warning("MultiprocessShuffledDataloader state is empty for global rank %d, expected key %s", self.config.global_rank, self._rank_id)
      return
    assert self.config.global_world_size == state_dict['world_size'], "global_world_size is inconsistent before and after checkpoint, dataloader resharding is not supported yet."
    state = pickle.loads(state_dict[self._rank_id])
    if 'dataset' in state and hasattr(self.dset, 'load_state_dict'):
      self.dset.load_state_dict(state['dataset'])

  def get_dummy_batch(self) -> Buffer:
    if self._dummy_batch is None:
      self._dummy_batch = attach_buffer_fields(self.shuffle_buffer_metadata['dummy_fields'])
    return self._dummy_batch

  def stats(self) -> ShuffleBufferStats:
    with self._r.pipeline(transaction=True) as pipe:
      full, empty = cast(
        list[int],
        pipe.scard(f'{self.queue_name}-full').scard(f'{self.queue_name}-empty').execute(),
      )
    in_flight = self.config.shuffle_size - full - empty
    return ShuffleBufferStats(full=full, empty=empty, in_flight=in_flight)

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

  def __del__(self) -> None:
    try:
      self._shutdown_workers()
    except Exception:
      pass

  def __iter__(self) -> Iterator[Buffer]:
    yielded = 0
    while True:
      if not self.ready_q.empty():
        buf, idx = self.ready_q.get()
        yield buf
        self.ready_e[idx].set()
        yielded += 1
        if self.max_iters is not None and yielded >= self.max_iters:
          return
      else:
        time.sleep(0.001)

      if time.perf_counter() - self.check_child_time > 1.0:
        self.check_children()
