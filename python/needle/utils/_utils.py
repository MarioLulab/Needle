import needle as ndl
import os
import multiprocess as mp
import queue
MP_STATUS_CHECK_INTERVAL = 5.0
r"""Interval (in seconds) to check status of processes to avoid hanging in
    multiprocessing data loading. This is mainly used in getting data from
    another process, in which case we need to periodically check whether the
    sender is alive to prevent hanging."""

r"""Dummy class used to resume the fetching when worker reuse is enabled"""
class _ResumeIteration(object):
    pass

r"""Dummy class used to signal the end of an IterableDataset"""
class _IterableDatasetStopIteration(object):
    worker_id: int


class ManagerWatchdog(object):  # type: ignore[no-redef]
    def __init__(self):
        self.manager_pid = os.getppid()
        self.manager_dead = False

    def is_alive(self):
        if not self.manager_dead:
            self.manager_dead = os.getppid() != self.manager_pid
        return not self.manager_dead

def _worker_loop(dataset, index_queue, data_queue, done_event, 
                 collate_fn, drop_last, worker_id, num_workers,
                 device, dtype):
    
    fetcher = ndl.data.DatasetFetcher(
        dataset=dataset,
        collate_fn=collate_fn,
        drop_last=drop_last,
        device=device,
        dtype=dtype
    )
    
    # When using Iterable mode, some worker can exit earlier than others due
    # to the IterableDataset behaving differently for different workers.
    # When such things happen, an `_IterableDatasetStopIteration` object is
    # sent over to the main process with the ID of this worker, so that the
    # main process won't send more tasks to this worker, and will send
    # `None` to this worker to properly exit it.
    #
    # Note that we cannot set `done_event` from a worker as it is shared
    # among all processes. Instead, we set the `iteration_end` flag to
    # signify that the iterator is exhausted. When either `done_event` or
    # `iteration_end` is set, we skip all processing step and just wait for
    # `None`.
    iteration_end = False
    
    watchdog = ManagerWatchdog()
    
    while watchdog.is_alive():
        try:
            r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue
        
        if isinstance(r, _ResumeIteration):
            # Acknowledge the main process
            data_queue.put((r, None))
            iteration_end = False
            # Recreate the fetcher for worker-reuse policy
            fetcher = ndl.data.DatasetFetcher(
                dataset=dataset,
                collate_fn=collate_fn,
                drop_last=drop_last,
                device=device,
                dtype=dtype
            )
            continue
        elif r is None:
            # Received the final signal
            assert done_event.is_set() or iteration_end
            break
        elif done_event.is_set() or iteration_end:
            # `done_event` is set. But I haven't received the final signal
            # (None) yet. I will keep continuing until get it, and skip the
            # processing steps.
            continue
        idx, index = r

        # data: _IterableDatasetStopIteration
        try:
            data = fetcher.fetch(index)
        except Exception as e:
            if isinstance(e, StopIteration):
                data = _IterableDatasetStopIteration(worker_id)
                # Set `iteration_end`
                #   (1) to save future `next(...)` calls, and
                #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                iteration_end = True
            else:
                raise e # TO-DO Exception Wrapper

        data_queue.put((idx, data))
        # data_queue.put((idx, None))
        del data, idx, index, r # save memory
        
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()