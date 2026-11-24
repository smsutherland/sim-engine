import base64
import json
import logging
import os
import re
import subprocess
import time
import typing as T
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import cloudpickle

from sim_engine.task import PyTask, Status, Task, TaskGenerator

LOGGER = logging.getLogger(__name__)


@dataclass()
class SlurmSpec:
    """A description of which nodes are available and the cpu count for each one.

    Automatically retreives the configuration for the current slurm allocation upon instantiation.

    Attributes
    ----------
    cpus_per_node: list[int]
        How many CPUs are on each allocated node
    hostnames: list[str]
        Name of each node allocated to the job
    """

    cpus_per_node: list[int]
    hostnames: list[str]

    def __init__(self):
        try:
            self.cpus_per_node = _parse_compound_number(
                os.environ["SLURM_JOB_CPUS_PER_NODE"]
            )

            self.hostnames = subprocess.check_output(
                ["scontrol", "show", "hostnames"], universal_newlines=True
            ).splitlines()
        except KeyError as e:
            e.add_note("We don't appear to be inside a slurm job!")
            raise


def _parse_compound_number(s: str) -> list[int]:
    r = []

    matches = re.findall(r"(\d+)(?:\(x(\d+)\))?", s)
    for n, repeats in matches:
        if repeats:
            r += [int(n)] * int(repeats)
        else:
            r.append(int(n))
    return r


class Scheduler:
    """Manages and executes a set of tasks provided to it within a slurm job allocation.

    Parameters
    ----------
    tasks: Sequence[Task] | None
        The initial set of tasks which will be managed by this Scheduler
    """

    tasks: list[Task]
    polling_period: float = 1.0
    max_tasks: int = 100

    def __init__(self, tasks: Sequence[Task] | None):
        if tasks:
            self.tasks = list(tasks)
        else:
            self.tasks = []
        self.slurm = SlurmSpec()

    def push_task(self, tasks: Task | Iterable[Task]):
        """Add a task or tasks to the current list kept by the Scheduler

        Parameters
        ----------
        tasks: Task | Iterable[Task]
            The task or tasks to add to the Scheduler
        """
        if isinstance(tasks, Task):
            self.tasks.append(tasks)
        else:
            self.tasks.extend(tasks)

    def run(self):
        """Start running the Scheduler's tasks"""
        self._check()

        n_jobs = 0
        had_task = False
        while n_jobs < self.max_tasks:
            if not had_task:
                # We sleep if the previous loop iteration didn't submit a task.
                # If it did, we'll try again just in case we can submit another one right away.
                time.sleep(self.polling_period)

            self._update_tasks()
            if self._done():
                break

            free_cpus = self._free_cpus()

            next_task = self._next_task()
            if next_task is None:
                if had_task:
                    LOGGER.debug("No tasks can run. Sleeping")
                had_task = False
                continue

            needed_free_cpus = next_task.num_cpus
            needed_nodes = _minimum_nodes_needed(
                needed_free_cpus, self.slurm.cpus_per_node
            )

            free_cpus_on_right_node_count = sum(sorted(free_cpus)[:needed_nodes])
            if free_cpus_on_right_node_count < needed_free_cpus:
                if had_task:
                    LOGGER.debug("Waiting for space. Sleeping.")
                had_task = False
                continue

            nodes_to_run_on = _argsort(free_cpus)[:needed_nodes]
            hostnames_to_run_on = [self.slurm.hostnames[i] for i in nodes_to_run_on]

            LOGGER.info(f"Starting task {next_task} on nodes {hostnames_to_run_on}")
            next_task.submit(hostnames_to_run_on)
            had_task = True
            n_jobs += 1

        LOGGER.info("All tasks have been started. Waiting for them to finish")
        while not self._done():
            self._update_tasks()
            time.sleep(self.polling_period * 4)

    def _check(self):
        cpus = sum(self.slurm.cpus_per_node)
        for t in self.tasks:
            if t.num_cpus > cpus:
                raise RuntimeError(
                    f"Task {t} requires {t.num_cpus} cpus, but allocation only has {cpus} cpus!"
                )
        # TODO: cyclic dependency checking

    def _next_task(self) -> Task | None:
        nt = None
        for t in self.tasks:
            if not t.pending:
                continue
            if nt is None:
                nt = t
                continue
            if t.priority > nt.priority:
                nt = t
        return nt

    def _update_tasks(self):
        for t in self.tasks:
            t._update_status(self)

    def _free_cpus(self) -> list[int]:
        cpus_dict = {}
        for t in self.tasks:
            if t.running:
                assert t.running_on is not None
                for n in t.running_on:
                    if n not in cpus_dict:
                        cpus_dict[n] = 0
                cpus_per_node = t.num_cpus // len(t.running_on)
                for n in t.running_on:
                    cpus_dict[n] += cpus_per_node
                for n in t.running_on[: t.num_cpus % len(t.running_on)]:
                    cpus_dict[n] += 1

        cpus = self.slurm.cpus_per_node.copy()
        for i, n in enumerate(self.slurm.hostnames):
            if n in cpus_dict:
                cpus[i] -= cpus_dict[n]
        return cpus

    def _done(self) -> bool:
        return all(t.done for t in self.tasks)

    def save(self, file: TextIO | Path | str):
        if isinstance(file, Path | str):
            with open(file) as f:
                self._save(f)
        else:
            self._save(file)

    def _save(self, f: TextIO):
        for t in self.tasks:
            if isinstance(t, TaskGenerator):
                kind = 2
            elif isinstance(t, PyTask):
                kind = 1
            elif isinstance(t, Task):
                kind = 0
            if isinstance(t, TaskGenerator | PyTask):
                func: bytes = cloudpickle.dumps(t.pyfunc)
                job = base64.b64encode(func).decode()
            else:
                job = t.job
            dep_indices = [self.tasks.index(d) for d in t.dependencies]
            json.dump(
                (
                    kind,
                    job,
                    t.cpus_per_task,
                    t.num_tasks,
                    t.status.value,
                    t.priority,
                    t.modules,
                    t.cwd,
                    dep_indices,
                ),
                f,
                separators=(",", ":"),
            )
            f.write("\n")

    @classmethod
    def load(cls, file: TextIO | Path | str) -> T.Self:
        if isinstance(file, Path | str):
            with open(file) as f:
                return cls._load(f)
        else:
            return cls._load(file)

    @classmethod
    def _load(cls, f: TextIO) -> T.Self:
        tasks = []
        deps = []
        for line in f:
            parsed: tuple[
                int,
                str | list[str],
                int,
                int,
                int,
                int,
                list[str],
                str | None,
                list[int],
            ] = json.loads(line)
            (
                kind,
                job,
                cpus_per_task,
                num_tasks,
                status_value,
                priority,
                modules,
                cwd,
                dep_indices,
            ) = parsed

            if kind == 0:
                task = Task(job, cpus_per_task, num_tasks, [], priority, cwd, modules)
                task.status = Status(status_value)
                tasks.append(task)
            elif kind == 1:
                assert isinstance(job, str)
                task = PyTask(
                    cloudpickle.loads(base64.b64decode(job.encode())),
                    cpus_per_task,
                    [],
                    priority,
                    cwd,
                    modules,
                )
                task.status = Status(status_value)
                tasks.append(task)
            elif kind == 2:
                assert isinstance(job, str)
                task = TaskGenerator(
                    cloudpickle.loads(base64.b64decode(job.encode())),
                    cpus_per_task,
                    [],
                    priority,
                    cwd,
                    modules,
                )
                task.status = Status(status_value)
                tasks.append(task)
            deps.append(dep_indices)

        # Reconstruct dependencies
        for t, d in zip(tasks, deps):
            t.dependencies = [tasks[i] for i in d]
        return cls(tasks)


def _minimum_nodes_needed(cpus: int, node_size: list[int]) -> int:
    node_size = sorted(node_size)
    i = 1
    while sum(node_size[:i]) < cpus:
        i += 1
    return i


def _argsort(l: list[int]) -> list[int]:
    return sorted(range(len(l)), key=l.__getitem__)
