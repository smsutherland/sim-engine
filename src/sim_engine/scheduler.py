import logging
import os
import re
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from sim_engine.task import AnyTask

LOGGER = logging.getLogger(__name__)

MAX_JOBS = 3

@dataclass()
class SlurmSpec:
    cpus_per_node: list[int]
    tasks_per_node: list[int]
    hostnames: list[str]

    def __init__(self):
        self.cpus_per_node = parse_compound_number(
            os.environ["SLURM_JOB_CPUS_PER_NODE"]
        )
        self.tasks_per_node = parse_compound_number(os.environ["SLURM_TASKS_PER_NODE"])

        self.hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames"], universal_newlines=True
        ).splitlines()


def parse_compound_number(s: str) -> list[int]:
    r = []

    matches = re.findall(r"(\d+)(?:\(x(\d+)\))?", s)
    for n, repeats in matches:
        if repeats:
            r += [int(n)] * int(repeats)
        else:
            r.append(int(n))
    return r


class Scheduler:
    tasks: list[AnyTask]
    polling_period: float = 1.0

    def __init__(self, tasks: Sequence[AnyTask] | None):
        if tasks:
            self.tasks = list(tasks)
        else:
            self.tasks = []
        self.slurm = SlurmSpec()

    def push_task(self, tasks: AnyTask | Iterable[AnyTask]):
        if isinstance(tasks, AnyTask):
            self.tasks.append(tasks)
        else:
            self.tasks.extend(tasks)

    def run(self):
        self.check()

        n_jobs = 0
        had_task = True
        while n_jobs < MAX_JOBS:
            if not had_task:
                # We sleep if the previous loop iteration didn't submit a task.
                # If it did, we'll try again just in case we can submit another one right away.
                time.sleep(self.polling_period)
            had_task = False

            self.update_tasks()
            if self.done():
                break

            free_cpus = self.free_cpus()

            next_task = self.next_task()
            if next_task is None:
                LOGGER.debug("No tasks can run. Sleeping")
                continue

            needed_free_cpus = next_task.num_cpus
            needed_nodes = minimum_nodes_needed(
                needed_free_cpus, self.slurm.cpus_per_node
            )

            free_cpus_on_right_node_count = sum(sorted(free_cpus)[:needed_nodes])
            if free_cpus_on_right_node_count < needed_free_cpus:
                LOGGER.debug("Waiting for space. Sleeping.")
                continue

            nodes_to_run_on = argsort(free_cpus)[:needed_nodes]
            hostnames_to_run_on = [self.slurm.hostnames[i] for i in nodes_to_run_on]

            LOGGER.info(f"Starting task {next_task} on nodes {hostnames_to_run_on}")
            next_task.submit(hostnames_to_run_on)
            had_task = True
            n_jobs += 1

    def check(self):
        return
        # TODO: TaskGenerators do not support asking for num_cpus right off the bat
        cpus = sum(self.slurm.cpus_per_node)
        for t in self.tasks:
            if t.num_cpus > cpus:
                raise RuntimeError(
                    f"Task {t} requires {t.num_cpus} cpus, but allocation only has {cpus} cpus!"
                )
        # TODO: cyclic dependency checking

    def next_task(self) -> AnyTask | None:
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

    def update_tasks(self):
        for t in self.tasks:
            t.update_status()

    def free_cpus(self) -> list[int]:
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

    def done(self) -> bool:
        return all(t.done for t in self.tasks)


def minimum_nodes_needed(cpus: int, node_size: list[int]) -> int:
    node_size = sorted(node_size)
    i = 1
    while sum(node_size[:i]) < cpus:
        i += 1
    return i


def argsort(l: list[int]) -> list[int]:
    return sorted(range(len(l)), key=l.__getitem__)
