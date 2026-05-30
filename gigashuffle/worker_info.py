from typing import Any

from torch.utils.data._utils import worker


def set_worker_info(dataset: Any, worker_id: int, num_workers: int, seed: int) -> None:
  worker._worker_info = worker.WorkerInfo(id=worker_id, num_workers=num_workers, seed=seed, dataset=dataset)
