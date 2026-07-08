from redis import StrictRedis
from redis.backoff import ExponentialWithJitterBackoff
from redis.retry import Retry


REDIS_RETRY_BASE_S = 0.01
REDIS_RETRY_CAP_S = 0.5
REDIS_RETRY_RETRIES = 12


def make_redis_client(host: str, port: int, db: int) -> StrictRedis:
  return StrictRedis(
    host=host,
    port=port,
    db=db,
    retry=Retry(
      ExponentialWithJitterBackoff(base=REDIS_RETRY_BASE_S, cap=REDIS_RETRY_CAP_S),
      REDIS_RETRY_RETRIES,
    ),
  )
