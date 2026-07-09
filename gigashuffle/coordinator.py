import hashlib
import logging
import os
import random
import time
from multiprocessing.connection import AuthenticationError, Client, Listener
from threading import Thread
from typing import Any


COORDINATOR_CONNECT_ERRORS = (AuthenticationError, EOFError, OSError)
COORDINATOR_ACCEPT_ERRORS = (AuthenticationError, EOFError, BrokenPipeError, ConnectionAbortedError, ConnectionResetError)
COORDINATOR_REQUEST_RETRIES = 10
COORDINATOR_REQUEST_RETRY_SLEEP_S = 0.1
logger = logging.getLogger(__name__)


def coordinator_authkey() -> bytes:
  return hashlib.blake2b(f'gigashuffle:{os.getuid()}'.encode(), digest_size=32).digest()


def coordinator_socket_path(queue_name: str) -> str:
  h = hashlib.sha1(f'gigashuffle:{os.getuid()}:{queue_name}'.encode()).hexdigest()[:24]
  return f'/tmp/gigashuffle-{os.getuid()}-{h}.sock'


class CoordinatorServer:
  def __init__(self, queue_name: str, attachment: Any, empty_indices: list[int]) -> None:
    self.queue_name = queue_name
    self.attachment = attachment
    self.empty = set(empty_indices)
    self.full: set[int] = set()
    self.attached = 0
    self.training_contexts: dict[str, bytes] = {}
    self.sock_path = coordinator_socket_path(queue_name)

  def start(self) -> Thread:
    try:
      os.unlink(self.sock_path)
    except FileNotFoundError:
      pass
    listener = Listener(self.sock_path, family='AF_UNIX', backlog=128, authkey=coordinator_authkey())
    t = Thread(target=self.serve, args=(listener,), name=f'gigashuffle-coordinator-{self.queue_name}', daemon=True)
    t.start()
    return t

  def pop(self, which: str, count: int) -> list[int]:
    q = self.full if which == 'full' else self.empty
    n = min(count, len(q))
    if n == 0:
      return []
    idxs = random.sample(tuple(q), n)
    q.difference_update(idxs)
    return idxs

  def push(self, which: str, idxs: list[int]) -> int:
    q = self.full if which == 'full' else self.empty
    before = len(q)
    q.update(map(int, idxs))
    return len(q) - before

  def stats(self) -> dict[str, int]:
    full = len(self.full)
    empty = len(self.empty)
    return dict(full=full, empty=empty, in_flight=self.attachment.metadata['shuffle_size'] - full - empty)

  def handle(self, req: dict[str, Any]) -> Any:
    op = req.get('op')
    if op == 'attach':
      if req.get('count_attach', True):
        self.attached += 1
      return self.attachment
    if op == 'attached_count':
      return self.attached
    if op == 'pop':
      return self.pop(req['which'], int(req['count']))
    if op == 'push':
      return self.push(req['which'], list(req['idxs']))
    if op == 'count':
      return len(self.full if req['which'] == 'full' else self.empty)
    if op == 'stats':
      return self.stats()
    if op == 'info':
      return dict(queue_name=self.queue_name, attached=self.attached, **self.stats())
    if op == 'evict':
      freed = set(map(int, req['idxs'])) & self.full
      self.full.difference_update(freed)
      self.empty.update(freed)
      return len(freed)
    if op == 'get_context':
      return self.training_contexts.get(req['rank_id'])
    if op == 'set_context':
      self.training_contexts[req['rank_id']] = req['context']
      return True
    raise RuntimeError(f'unknown coordinator op {op!r}')

  def serve(self, listener: Listener) -> None:
    try:
      while True:
        try:
          conn = listener.accept()
        except COORDINATOR_ACCEPT_ERRORS:
          continue
        try:
          try:
            result = self.handle(conn.recv())
            conn.send(dict(ok=True, result=result))
          except Exception as e:
            conn.send(dict(ok=False, error=str(e)))
        except (EOFError, OSError):
          pass
        finally:
          conn.close()
    finally:
      listener.close()
      try:
        os.unlink(self.sock_path)
      except FileNotFoundError:
        pass


class CoordinatorClient:
  def __init__(self, queue_name: str, retries: int = COORDINATOR_REQUEST_RETRIES) -> None:
    self.sock_path = coordinator_socket_path(queue_name)
    self.retries = retries

  @classmethod
  def from_socket_path(cls, sock_path: str, retries: int = COORDINATOR_REQUEST_RETRIES) -> 'CoordinatorClient':
    client = cls.__new__(cls)
    client.sock_path = sock_path
    client.retries = retries
    return client

  def connect(self):
    for attempt in range(self.retries + 1):
      try:
        return Client(self.sock_path, family='AF_UNIX', authkey=coordinator_authkey())
      except COORDINATOR_CONNECT_ERRORS as e:
        if attempt >= self.retries:
          raise
        logger.warning(f"retrying gigashuffle coordinator connection to {self.sock_path} after {type(e).__name__}: {e}")
        time.sleep(COORDINATOR_REQUEST_RETRY_SLEEP_S)
    raise RuntimeError("failed to connect to gigashuffle coordinator")

  def request(self, op: str, **kwargs) -> Any:
    conn = self.connect()
    try:
      conn.send(dict(op=op, **kwargs))
      msg = conn.recv()
    finally:
      conn.close()
    if not isinstance(msg, dict) or not msg.get('ok'):
      raise RuntimeError(f"gigashuffle coordinator request {op!r} failed: {msg}")
    return msg['result']

  def attach(self, count_attach: bool = True) -> Any:
    return self.request('attach', count_attach=count_attach)

  def attached_count(self) -> int:
    return int(self.request('attached_count'))

  def pop(self, which: str, count: int) -> list[int]:
    return [int(i) for i in self.request('pop', which=which, count=count)]

  def push(self, which: str, idxs: list[int]) -> int:
    if not idxs:
      return 0
    return int(self.request('push', which=which, idxs=idxs))

  def count(self, which: str) -> int:
    return int(self.request('count', which=which))

  def stats(self) -> dict[str, int]:
    return self.request('stats')

  def info(self) -> dict[str, int | str]:
    return self.request('info')

  def evict(self, idxs: list[int]) -> int:
    return int(self.request('evict', idxs=idxs))

  def get_context(self, rank_id: str) -> bytes | None:
    return self.request('get_context', rank_id=rank_id)

  def set_context(self, rank_id: str, context: bytes) -> None:
    self.request('set_context', rank_id=rank_id, context=context)
