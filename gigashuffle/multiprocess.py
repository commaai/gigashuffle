import hashlib
import logging
import math
import os
import pickle
import random
import re
import sys
import time
import torch
import numpy as np
import torch.multiprocessing as mp
from itertools import batched, count
from multiprocessing.synchronize import Event
from multiprocessing.queues import SimpleQueue
from typing import Any, Iterator, cast
from redis import StrictRedis
from setproctitle import setproctitle
from torch.utils.data import Dataset, IterableDataset
from gigashuffle.worker_info import set_worker_info
from gigashuffle.config import DataloaderConfig


Buffer = list[dict[str, torch.Tensor]]
CHUNK_SIZE = 1024*64
LOG_INTERVAL_S = 5.0
ShuffleBufferMetadata = dict[str, Any]
logger = logging.getLogger(__name__)
already_warned = False

# NOTE: For high-throughput tasks, calling torch.set_num_threads(1) seems to significantly reduce CPU usage in the gigashuffle writers.
# This SO thread about a similar issue notes that it "was caused by a bad interaction of OpenMP and multiprocessing".
# https://stackoverflow.com/questions/65057388/pytorch-multiprocessing-with-shared-memory-causes-matmul-to-be-30x-slower-with
torch.set_num_threads(1)


class BatchSizeMismatch(Exception):
  pass


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


def get_batch_from_input_samples(input_samples, input_bs, bs):
  idxs = (np.arange(bs) % input_bs).tolist()
  return [{k:torch.as_tensor(v[idxs]) for k,v in input_samples[i].items()} for i in range(len(input_samples))]


def fetch_initial_sample(dset: Any, config: DataloaderConfig) -> tuple[Buffer, int, tuple[int, str]]:
  shuffle_size = config.shuffle_size
  min_mixing_n = int(config.min_mixing * shuffle_size)
  input_samples, input_bs, input_bs_key = get_samples(dset, max_retries=config.writer_max_retries)
  memory_size = get_memory_size(input_samples, input_bs)

  logger.info("each element uses %d bytes, total buffer size is %.3fgb", memory_size, shuffle_size*memory_size/1e9)
  if shuffle_size < config.bs * config.local_world_size:
    N = config.local_world_size * config.num_readers
    raise RuntimeError(f"Shuffle buffer must be large enough to accommodate at least N batches, but buffer size = {shuffle_size}, batch size = {config.bs}, N = {N}")
  if min_mixing_n >= shuffle_size - 2*input_bs:
    raise RuntimeError(f"To avoid deadlock, min_mixing_n ({min_mixing_n}) must be less than {shuffle_size - 2*input_bs}")

  return input_samples, input_bs, input_bs_key


def initialize_redis_queue(r: StrictRedis, queue_name: str, shuffle_size: int) -> None:
  logger.info("setting up %s on redis version %s", queue_name, r.execute_command('INFO')['redis_version'])
  r.delete(f"{queue_name}-shared-buffer-meta")
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


def create_named_shuffle_buffer(first_samples: Buffer, shuffle_size: int, input_bs: int, input_bs_key: tuple[int, str], queue_name: str, shm_dir: str, print_shapes: bool = True) -> tuple[Buffer, ShuffleBufferMetadata]:
  os.makedirs(shm_dir, exist_ok=True)
  safe_queue_name = re.sub(r'[^A-Za-z0-9_.-]', '_', queue_name)
  shuffle_buffer_metadata: ShuffleBufferMetadata = dict(queue_name=queue_name, shuffle_size=shuffle_size, input_bs=input_bs, input_bs_key=input_bs_key, fields=[])
  shuffle_buffer = []

  for i in range(len(first_samples)):
    b = {}
    for k,v in first_samples[i].items():
      dtype = numpy_type_to_torch(v.dtype)
      shape = tuple([shuffle_size]+list(v.shape[1:]))
      digest = hashlib.sha256(f'{i}:{k}'.encode('utf-8')).hexdigest()[:16]
      path = os.path.join(shm_dir, f'gigashuffle-{safe_queue_name}-{i}-{digest}.bin')
      fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
      try:
        os.ftruncate(fd, math.prod(shape) * torch.empty((), dtype=dtype).element_size())
      finally:
        os.close(fd)
      if print_shapes:
        logger.info("allocating shape %s for %s with type %s at %s", list(shape), k, dtype, path)
      b[k] = torch.from_file(path, shared=True, size=math.prod(shape), dtype=dtype).view(shape)
      shuffle_buffer_metadata['fields'].append(dict(i=i, k=k, shape=shape, dtype=str(dtype).removeprefix('torch.'), path=path))
    shuffle_buffer.append(b)
  return shuffle_buffer, shuffle_buffer_metadata


def attach_named_shuffle_buffer(shuffle_buffer_metadata: ShuffleBufferMetadata, bs: int | None = None, shared: bool = False) -> Buffer:
  shuffle_buffer: Buffer = [{} for _ in range(max(t['i'] for t in shuffle_buffer_metadata['fields']) + 1)]
  for t in shuffle_buffer_metadata['fields']:
    dtype = getattr(torch, t['dtype'])
    shape = tuple(([bs] if bs is not None else [t['shape'][0]]) + list(t['shape'][1:]))
    if bs is None:
      shuffle_buffer[t['i']][t['k']] = torch.from_file(t['path'], shared=True, size=math.prod(shape), dtype=dtype).view(shape)
    else:
      tensor = torch.empty(shape, dtype=dtype)
      shuffle_buffer[t['i']][t['k']] = tensor.share_memory_() if shared else tensor
  return shuffle_buffer


def wait_for_shuffle_buffer_metadata(r: StrictRedis, queue_name: str) -> ShuffleBufferMetadata:
  while True:
    raw = cast(bytes | None, r.get(f'{queue_name}-shared-buffer-meta'))
    if raw is not None:
      shuffle_buffer_metadata = cast(ShuffleBufferMetadata, pickle.loads(raw))
      if all(os.path.exists(t['path']) for t in shuffle_buffer_metadata['fields']):
        return shuffle_buffer_metadata
    time.sleep(0.1)


def writer(dset: Dataset, config: DataloaderConfig, proc_idx: int, queue_name: str) -> None:
  init_logger()
  setproctitle('gigashuffle writer %d %d' % (config.local_rank, proc_idx))
  os.system('renice -n 3 -p %d > /dev/null' % os.getpid())

  shuffle_size = config.shuffle_size
  global_proc_idx = config.global_rank * config.num_writers + proc_idx
  total_procs = config.global_world_size * config.num_writers
  local_proc_idx = config.local_rank * config.num_writers + proc_idx
  random.seed(global_proc_idx)
  torch.manual_seed(global_proc_idx)
  set_worker_info(dset, worker_id=global_proc_idx, num_workers=total_procs, seed=global_proc_idx)

  r = StrictRedis(host=config.redis_host, port=config.redis_port, db=config.redis_db)
  dset_iter = iter(dset) if hasattr(dset, '__iter__') else dset
  if local_proc_idx == 0:
    initialize_redis_queue(r, queue_name, shuffle_size)
    input_samples, input_bs, input_bs_key = fetch_initial_sample(dset_iter, config)
    shuffle_buffer, shuffle_buffer_metadata = create_named_shuffle_buffer(input_samples, shuffle_size, input_bs, input_bs_key, queue_name, config.shm_dir)
    initial_idx_list = list(range(input_bs))
    r.srem(f'{queue_name}-empty', *initial_idx_list)
    for i in range(len(shuffle_buffer)):
      for k in shuffle_buffer[i].keys():
        tmp = torch.as_tensor(input_samples[i][k])
        if tmp.device != shuffle_buffer[i][k].device or tmp.dtype != shuffle_buffer[i][k].dtype:
          tmp = tmp.to(device=shuffle_buffer[i][k].device, dtype=shuffle_buffer[i][k].dtype)
        shuffle_buffer[i][k][initial_idx_list] = tmp
    r.sadd(f'{queue_name}-full', *initial_idx_list)
    r.set(f'{queue_name}-shared-buffer-meta', pickle.dumps(shuffle_buffer_metadata))
  else:
    shuffle_buffer_metadata = wait_for_shuffle_buffer_metadata(r, queue_name)
    shuffle_buffer = attach_named_shuffle_buffer(shuffle_buffer_metadata)

  logger.info("writer %d-%d initialized with input_bs %d output_bs %d", config.global_rank, proc_idx, shuffle_buffer_metadata['input_bs'], config.bs)
  while True:
    samples, local_input_bs, _ = get_samples(dset_iter, shuffle_buffer_metadata['input_bs_key'], max_retries=config.writer_max_retries)
    max_input_bs = (shuffle_size - config.bs) // (config.local_world_size * config.num_writers)
    if local_input_bs > max_input_bs:
      local_input_bs = max_input_bs
      print_small_shuffle_warning()

    idx_list = fetch_rand_from_queue(r, f'{queue_name}-empty', local_input_bs)
    for i in range(len(shuffle_buffer)):
      for k in shuffle_buffer[i].keys():
        tmp = torch.as_tensor(samples[i][k])[:local_input_bs]
        if tmp.device != shuffle_buffer[i][k].device or tmp.dtype != shuffle_buffer[i][k].dtype:
          tmp = tmp.to(device=shuffle_buffer[i][k].device, dtype=shuffle_buffer[i][k].dtype)
        shuffle_buffer[i][k][idx_list] = tmp
    r.sadd(f'{queue_name}-full', *idx_list)


def reader(config: DataloaderConfig, ready_q: SimpleQueue[tuple[Buffer, int]], ready_e: Event, proc_idx: int, queue_name: str):
  init_logger()
  setproctitle('gigashuffle reader %d %d' % (config.local_rank, proc_idx))
  r = StrictRedis(host=config.redis_host, port=config.redis_port, db=config.redis_db)
  min_mixing_n = int(config.min_mixing * config.shuffle_size)
  shuffle_buffer_metadata = wait_for_shuffle_buffer_metadata(r, queue_name)
  shuffle_buffer = attach_named_shuffle_buffer(shuffle_buffer_metadata)
  reader_buffer = attach_named_shuffle_buffer(shuffle_buffer_metadata, bs=config.bs, shared=True)

  for batch_idx in count():
    idx_list = fetch_rand_from_queue(r, f'{queue_name}-full', config.bs, min_mixing_n=min_mixing_n, log_progress=batch_idx == 0 and config.local_rank == 0 and proc_idx == 0)
    for buffer_idx in range(len(shuffle_buffer)):
      for k in shuffle_buffer[buffer_idx].keys():
        reader_buffer[buffer_idx][k][:] = shuffle_buffer[buffer_idx][k][idx_list]
    r.sadd(f'{queue_name}-empty', *idx_list)

    ready_q.put((reader_buffer, proc_idx))
    while not ready_e.is_set():
      ready_e.wait()
    ready_e.clear()


class MultiprocessShuffledDataloader(IterableDataset):
  def __init__(self, dset: Dataset, config: DataloaderConfig) -> None:
    init_logger()
    self.dset = dset
    self.config = config
    assert config.num_writers > 0, "gigashuffle requires num_writers > 0"
    assert config.queue_name, "MultiprocessShuffledDataloader requires config.queue_name"
    self.queue_name = config.queue_name
    self._rank_id = f'global_rank_{config.global_rank}'
    self._closed = False

    r = StrictRedis(host=config.redis_host, port=config.redis_port, db=config.redis_db)
    try:
      assert r.ping()
    except Exception as e:
      raise AssertionError(f"Redis is not reachable at {config.redis_host}:{config.redis_port}/{config.redis_db}") from e

    ctx = mp.get_context('spawn')
    self.ready_q: SimpleQueue[tuple[Buffer, int]] = ctx.SimpleQueue()
    self.ready_e = [ctx.Event() for _ in range(self.config.num_readers)]
    self.children = []
    self.check_child_time = 0.

    for i in range(self.config.num_readers):
      self.children.append(ctx.Process(target=reader, args=(config, self.ready_q, self.ready_e[i], i, self.queue_name), daemon=True))
    for i in range(self.config.num_writers):
      self.children.append(ctx.Process(target=writer, args=(dset, config, i, self.queue_name), daemon=True))

    for i, p in enumerate(self.children):
      p.start()
    self.check_children()
    self.shuffle_buffer_metadata = wait_for_shuffle_buffer_metadata(r, self.queue_name)

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

  def get_dummy_batch(self, bs: int | None = None) -> Buffer:
    bs = self.config.bs if bs is None else bs
    shuffle_buffer = attach_named_shuffle_buffer(self.shuffle_buffer_metadata)
    return get_batch_from_input_samples(shuffle_buffer, self.shuffle_buffer_metadata['input_bs'], bs)

  def check_children(self) -> None:
    for p in self.children:
      if not p.is_alive():
        raise RuntimeError(f"MultiprocessShuffledDataloader child {p.name} (pid={p.pid}) died (exitcode={p.exitcode}). Aborting.")
    self.check_child_time = time.perf_counter()

  def close(self, unlink_shared_memory: bool = True) -> None:
    if self._closed:
      return
    self._closed = True
    for p in self.children:
      if p.is_alive():
        p.terminate()
    for p in self.children:
      p.join(timeout=5)
    if unlink_shared_memory and self.config.local_rank == 0:
      for t in self.shuffle_buffer_metadata['fields']:
        try:
          os.unlink(t['path'])
        except FileNotFoundError:
          pass
      r = StrictRedis(host=self.config.redis_host, port=self.config.redis_port, db=self.config.redis_db)
      r.delete(f'{self.queue_name}-shared-buffer-meta')

  def __iter__(self) -> Iterator[Buffer]:
    while True:
      if not self.ready_q.empty():
        buf, idx = self.ready_q.get()
        yield buf
        self.ready_e[idx].set()
      else:
        time.sleep(0.001)

      if time.perf_counter() - self.check_child_time > 1.0:
        self.check_children()
