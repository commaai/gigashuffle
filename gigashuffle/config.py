from dataclasses import dataclass


@dataclass
class DataloaderConfig:
  bs: int
  shuffle_size: int
  min_mixing: float = 0.5
  num_writers: int = 2
  num_readers: int = 2
  writer_max_retries: int = 100
  fill_once: bool = False
  evict_on_read: bool = True
  local_rank: int = 0
  global_rank: int = 0
  local_world_size: int = 1
  global_world_size: int = 1
  queue_name: str = ''
  # Per-rank writer sharding. num_writers is the number of writer processes spawned by THIS
  # rank (may be 0). The three fields below describe the writer set across ALL ranks and must be
  # identical on every rank. When left at their sentinels they are derived assuming a uniform
  # num_writers on every rank (the legacy behavior), so existing callers are unaffected.
  total_writers: int = 0  # 0 => global_world_size * num_writers
  global_writer_offset: int = -1  # -1 => global_rank * num_writers
  local_writer_offset: int = -1  # -1 => local_rank * num_writers

  def resolved_total_writers(self) -> int:
    return self.total_writers or (self.global_world_size * self.num_writers)

  def resolved_global_writer_offset(self) -> int:
    return self.global_writer_offset if self.global_writer_offset >= 0 else (self.global_rank * self.num_writers)

  def resolved_local_writer_offset(self) -> int:
    return self.local_writer_offset if self.local_writer_offset >= 0 else (self.local_rank * self.num_writers)
