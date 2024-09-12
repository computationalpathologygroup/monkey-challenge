import multiprocessing
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Process

import psutil


class PredictionProcessingError(Exception):
    def __init__(self, prediction, error):
        self.prediction = prediction
        self.error = error

    def __str__(self):
        return f"Error for prediction {self.prediction}: {self.error}"


def get_max_workers():
    """
    Returns the maximum number of concurrent workers

    The optimal number of workers ultimately depends on how many resources
    each process will call upon.

    To limit this, update the Dockerfile GRAND_CHALLENGE_MAX_WORKERS
    """

    environ_cpu_limit = os.getenv("GRAND_CHALLENGE_MAX_WORKERS")
    cpu_count = multiprocessing.cpu_count()
    return min(
        [
            int(environ_cpu_limit or cpu_count),
            cpu_count,
        ]
    )


def run_prediction_processing(*, fn, predictions):
    """
    Processes predictions in a separate process.

    This takes child processes into account:
    - if any child process is terminated, all prediction processing will abort
    - after prediction processing is done, all child processes are terminated

    Parameters
    ----------
    fn : function
        Function to execute that will process each prediction

    predictions : list
        List of predictions.

    Returns
    -------
    A list of results
    """
    with Manager() as manager:
        results = manager.list()
        errors = manager.list()

        pool_worker = _start_pool_worker(
            fn=fn,
            predictions=predictions,
            max_workers=get_max_workers(),
            results=results,
            errors=errors,
        )
        try:
            pool_worker.join()
        finally:
            pool_worker.terminate()

        for prediction, e in errors:
            raise PredictionProcessingError(
                prediction=prediction,
                error=e,
            ) from e

        return list(results)


def _start_pool_worker(fn, predictions, max_workers, results, errors):
    process = Process(
        target=_pool_worker,
        name="PredictionProcessing",
        kwargs=dict(
            fn=fn,
            predictions=predictions,
            max_workers=max_workers,
            results=results,
            errors=errors,
        ),
    )
    process.start()

    return process


def _pool_worker(*, fn, predictions, max_workers, results, errors):
    terminating_child_processes = False

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        try:

            def handle_error(error, prediction="Unknown"):
                executor.shutdown(wait=False, cancel_futures=True)
                errors.append((prediction, error))

                nonlocal terminating_child_processes
                terminating_child_processes = True
                _terminate_child_processes()

            def sigchld_handler(*_, **__):
                if not terminating_child_processes:
                    handle_error(
                        RuntimeError(
                            "Child process was terminated unexpectedly"
                        )
                    )

            # Register the SIGCHLD handler
            signal.signal(signal.SIGCHLD, sigchld_handler)

            # Submit the processing tasks of the predictions
            futures = [
                executor.submit(fn, prediction) for prediction in predictions
            ]
            future_to_predictions = {
                future: item
                for future, item in zip(futures, predictions, strict=True)
            }

            for future in as_completed(future_to_predictions):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    handle_error(e, prediction=future_to_predictions[future])
        finally:
            terminating_child_processes = True
            _terminate_child_processes()


def _terminate_child_processes():
    current_process = psutil.Process(os.getpid())
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass  # Not a problem

    # Wait for processes to terminate
    gone, still_alive = psutil.wait_procs(children, timeout=5)

    # Forcefully kill any remaining processes
    for p in still_alive:
        print(f"Forcefully killing child process {p.pid}")
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass  # That is fine
