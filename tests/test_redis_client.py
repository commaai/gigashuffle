import pytest
import redis.backoff
import redis.retry
from redis.exceptions import ConnectionError

from gigashuffle.redis_client import (
  REDIS_RETRY_BASE_S,
  REDIS_RETRY_CAP_S,
  REDIS_RETRY_RETRIES,
  make_redis_client,
)


def test_make_redis_client_retries_transient_connection_errors(monkeypatch):
  client = make_redis_client(host='localhost', port=6379, db=6)
  retry = client.connection_pool.connection_kwargs['retry']
  sleep_delays = []
  monkeypatch.setattr(redis.backoff.random, 'random', lambda: 1.0)
  monkeypatch.setattr(redis.retry, 'sleep', lambda delay: sleep_delays.append(delay))

  attempts = 0

  def flaky_command():
    nonlocal attempts
    attempts += 1
    if attempts <= REDIS_RETRY_RETRIES:
      raise ConnectionError('temporary redis failure')
    return 'ok'

  assert retry.call_with_retry(flaky_command, lambda error: None) == 'ok'
  assert attempts == REDIS_RETRY_RETRIES + 1
  assert sleep_delays == [
    min(REDIS_RETRY_CAP_S, REDIS_RETRY_BASE_S * 2**failure)
    for failure in range(1, REDIS_RETRY_RETRIES + 1)
  ]

  non_retryable_attempts = 0

  def non_retryable_command():
    nonlocal non_retryable_attempts
    non_retryable_attempts += 1
    raise ValueError('not a redis connection error')

  with pytest.raises(ValueError):
    retry.call_with_retry(non_retryable_command, lambda error: None)
  assert non_retryable_attempts == 1
  assert len(sleep_delays) == REDIS_RETRY_RETRIES
