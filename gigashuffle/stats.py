#!/usr/bin/env python3

import argparse
import os
from glob import glob

from gigashuffle.coordinator import COORDINATOR_CONNECT_ERRORS, CoordinatorClient


DEFAULT_STATS_DIR = '/tmp'
STATS_ERRORS = COORDINATOR_CONNECT_ERRORS + (RuntimeError,)


def live_stats(stats_dir: str = DEFAULT_STATS_DIR) -> list[dict[str, int | str]]:
  stats = []
  seen_queues = set()
  socket_paths = sorted(glob(os.path.join(stats_dir, f'gigashuffle-{os.getuid()}-*.sock')))
  for sock_path in socket_paths:
    try:
      info = CoordinatorClient.from_socket_path(sock_path, retries=0).info()
    except STATS_ERRORS:
      continue
    if info['queue_name'] in seen_queues:
      continue
    seen_queues.add(info['queue_name'])
    stats.append(info)
  return stats


def print_stats(stats_dir: str = DEFAULT_STATS_DIR) -> None:
  print("queue_name full empty in_flight attached")
  for s in live_stats(stats_dir):
    print(f"{s['queue_name']} {s['full']} {s['empty']} {s['in_flight']} {s['attached']}")


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('stats_dir', nargs='?', default=DEFAULT_STATS_DIR)
  args = parser.parse_args(argv)
  print_stats(args.stats_dir)


if __name__ == "__main__":
  main()
