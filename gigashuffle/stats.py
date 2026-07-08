#!/usr/bin/env python3

from glob import glob

from gigashuffle.coordinator import COORDINATOR_CONNECT_ERRORS, CoordinatorClient, coordinator_socket_glob


STATS_ERRORS = COORDINATOR_CONNECT_ERRORS + (RuntimeError,)


def live_stats() -> list[dict[str, int | str]]:
  stats = []
  for sock_path in sorted(glob(coordinator_socket_glob())):
    try:
      stats.append(CoordinatorClient.from_socket_path(sock_path).info())
    except STATS_ERRORS:
      pass
  return stats


def print_stats() -> None:
  print("queue_name full empty in_flight attached")
  for s in live_stats():
    print(f"{s['queue_name']} {s['full']} {s['empty']} {s['in_flight']} {s['attached']}")


def main() -> None:
  print_stats()


if __name__ == "__main__":
  main()
