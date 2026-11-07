import enum
import logging
import os
import re
import shlex
import subprocess
import time
import typing as T
from collections.abc import Iterable
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass
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


class Status(enum.Enum):
    Waiting = 0  # Waiting for dependencies
    Pending = 1  # Could run, but hasn't
    Running = 2  # Actively running
    Succeeded = 3  # Ran and succeeded
    Failed = 4  # Ran and failed
    DepFailed = 5  # Dependency failed


@dataclass
class Task:
    job: str
    cpus_per_task: int
    num_tasks: int
    dependencies: list[T.Self]
    status: Status
    direct: bool = False
    priority: int = 0
    child: subprocess.Popen | None = None
    ran_on: list[str] | None = None

    def __init__(
        self,
        job: str,
        cpus_per_task: int,
        num_tasks: int,
        dependencies: list[T.Self] | None = None,
        # direct: bool = False,
    ):
        self.job = job
        self.cpus_per_task = cpus_per_task
        self.num_tasks = num_tasks
        self.dependencies = dependencies or []
        # self.direct = direct
        self.direct = True
        if self.dependencies:
            self.status = Status.Waiting
        else:
            self.status = Status.Pending

    def update_status(self):
        for d in self.dependencies:
            d.update_status()
        match self.status:
            case Status.Waiting:
                if all(d.succeeded for d in self.dependencies):
                    self.status = Status.Pending
                elif any(d.failed or d.dep_failed for d in self.dependencies):
                    self.status = Status.DepFailed
            case Status.Running:
                if self.child is None:
                    raise RuntimeError(
                        "Running task doesn't have an srun child process"
                    )
                if (retval := self.child.poll()) is not None:
                    if retval == 0:
                        self.status = Status.Succeeded
                    else:
                        self.status = Status.Failed

    def submit(self, workers: list[str]) -> subprocess.Popen:
        workers = sorted(workers)
        undone_deps = [d for d in self.dependencies if not d.succeeded]
        if undone_deps:
            raise RuntimeError(
                f"Trying to submit task: {self} when the following tasks haven't completed:\n{undone_deps}"
            )

        if self.direct:
            cmd = [
                "srun",
                f"--ntasks={self.num_tasks}",
                f"--cpus-per-task={self.cpus_per_task}",
                "--nodelist=" + ",".join(workers),
                "--exact",
                "--cpu-bind=cores",
                *shlex.split(self.job),
            ]
        else:
            raise NotImplementedError("Indirect srun calls not yet supported")

        LOGGER.debug(" ".join(cmd))
        self.ran_on = workers
        self.child = subprocess.Popen(cmd)
        self.status = Status.Running
        return self.child

    @property
    def num_cpus(self) -> int:
        return self.cpus_per_task * self.num_tasks

    @property
    def waiting(self) -> bool:
        return self.status == Status.Waiting

    @property
    def pending(self) -> bool:
        return self.status == Status.Pending

    @property
    def running(self) -> bool:
        return self.status == Status.Running

    @property
    def succeeded(self) -> bool:
        return self.status == Status.Succeeded

    @property
    def failed(self) -> bool:
        return self.status == Status.Failed

    @property
    def finished(self) -> bool:
        return self.succeeded or self.failed

    @property
    def dep_failed(self) -> bool:
        return self.status == Status.DepFailed

    @property
    def done(self) -> bool:
        return self.finished or self.dep_failed


class Scheduler:
    tasks: list[Task]
    polling_period: float = 1.0

    def __init__(self, tasks: list[Task] | None):
        self.tasks = tasks or []
        self.slurm = SlurmSpec()

    def push_task(self, tasks: Task | Iterable[Task]):
        if isinstance(tasks, Task):
            self.tasks.append(tasks)
        else:
            self.tasks.extend(tasks)

    def run(self):
        self.check()

        had_task = True
        while True:
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

    def check(self):
        cpus = sum(self.slurm.cpus_per_node)
        for t in self.tasks:
            if t.num_cpus > cpus:
                raise RuntimeError(
                    f"Task {t} requires {t.num_cpus} cpus, but allocation only has {cpus} cpus!"
                )
        # TODO: cyclic dependency checking

    def next_task(self) -> Task | None:
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
                assert t.ran_on is not None
                for n in t.ran_on:
                    if n not in cpus_dict:
                        cpus_dict[n] = 0
                cpus_per_node = t.num_cpus // len(t.ran_on)
                for n in t.ran_on:
                    cpus_dict[n] += cpus_per_node
                for n in t.ran_on[: t.num_cpus % len(t.ran_on)]:
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
