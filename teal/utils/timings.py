from collections import defaultdict
import os
import time
from typing import DefaultDict, List, Optional, Union
import numpy as np
from tabulate import tabulate

import torch


class CPUTimer:
    def __init__(self):
        self.start: Optional[float] = None
        self.end: Optional[float] = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.end = time.time()

    def time_elapsed_ms(self):
        assert self.start is not None, "Scope not entered. "
        assert self.end is not None, "Scope not exited. "
        return (self.end - self.start) * 1000


class GPUTimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()  # type: ignore

    def __exit__(self, *args):
        self.end.record()  # type: ignore

    def time_elapsed_ms(self):
        torch.cuda.synchronize()
        return self.start.elapsed_time(self.end)


class Timings:
    def __init__(self):
        self.timing_data: DefaultDict[
            str, List[Union[CPUTimer, GPUTimer]]
        ] = defaultdict(list)

    def scope(self, name: str, on_gpu: bool = False):
        scope = GPUTimer() if on_gpu and torch.cuda.is_available() else CPUTimer()
        self.timing_data[name].append(scope)
        return scope

    def print(self, step: int):
        timing_data = []
        for name, scopes in self.timing_data.items():
            elapsed_times = np.array([scope.time_elapsed_ms() for scope in scopes])
            mean_time = np.mean(elapsed_times)
            median_time = np.median(elapsed_times)
            q95_time = np.quantile(elapsed_times, 0.95)
            timing_data.append(
                (name, mean_time, median_time, q95_time, len(elapsed_times))
            )

        self.timing_data.clear()

        info_data = [
            ["Step", step],
            ["Timestamp", time.strftime("%Y-%m-%d %H:%M:%S")],
            ["Git commit", os.popen("git rev-parse HEAD").read().strip()],
        ]

        if torch.cuda.is_available():
            gpu_mem_free, gpu_mem_total = torch.cuda.mem_get_info()
            gpu_mem_usage = gpu_mem_total - gpu_mem_free
            gpu_mem_percent = gpu_mem_usage / gpu_mem_total * 100
            info_data.append(
                [
                    "GPU memory usage",
                    f"{gpu_mem_usage / (1024 ** 3):.2f} GB of {gpu_mem_total / (1024 ** 3):.2f} GB ({gpu_mem_percent:.2f}%)",
                ]
            )

        print(
            tabulate(
                tabular_data=info_data,
                tablefmt="fancy_grid",
                colalign=("left", "right"),
            )
        )

        print(
            tabulate(
                tabular_data=timing_data,
                tablefmt="fancy_grid",
                headers=["Scope name", "Mean", "Median", "95th percentile", "Count"],
            )
        )


_DEFAULT_TIMINGS = Timings()


def timed_scope(name: str, on_gpu: bool = False):
    return _DEFAULT_TIMINGS.scope(name, on_gpu)


def print_timings_summary(step: int):
    _DEFAULT_TIMINGS.print(step)
