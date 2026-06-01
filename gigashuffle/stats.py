#!/usr/bin/env python3
import pickle

from redis import StrictRedis


def print_stats(host="localhost", port=6379, db=6):
  r = StrictRedis(host=host, port=port, db=db)
  queues = set()

  for raw_key in r.scan_iter("gigashuffle-*-shared-buffer-meta"):
    key = raw_key.decode()
    queues.add(key.removesuffix("-shared-buffer-meta"))

  for raw_key in r.scan_iter("gigashuffle-*-initializing"):
    key = raw_key.decode()
    queues.add(key.removesuffix("-initializing"))

  for queue in sorted(queues):
    raw_meta = r.get(f"{queue}-shared-buffer-meta")
    print(queue)
    if raw_meta is None:
      print("  initializing: 1")
      continue

    meta = pickle.loads(raw_meta)
    full = r.scard(f"{queue}-full")
    empty = r.scard(f"{queue}-empty")
    size = meta["shuffle_size"]
    attached = int(r.get(f"{queue}-shared-buffer-attached") or 0)
    print(f"  full: {full}")
    print(f"  empty: {empty}")
    print(f"  inflight: {size-full-empty}")
    print(f"  size: {size}")
    print(f"  attached: {attached}")


def main():
  print_stats()


if __name__ == "__main__":
  main()
