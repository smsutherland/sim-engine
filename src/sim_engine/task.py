import logging
import shlex
import subprocess
import sys
import typing as T
from enum import Enum

import cloudpickle

LOGGER = logging.getLogger(__name__)


class Status(Enum):
    Waiting = 0  # Waiting for dependencies
    Pending = 1  # Could run, but hasn't
    Running = 2  # Actively running
    Succeeded = 3  # Ran and succeeded
    Failed = 4  # Ran and failed
    DepFailed = 5  # Dependency failed


class Task:
    job: list[str]
    cpus_per_task: int
    num_tasks: int
    dependencies: list[T.Self]
    status: Status
    priority: int = 0
    child: subprocess.Popen | None = None
    running_on: list[str] | None = None

    def __init__(
        self,
        job: str | list[str],
        cpus_per_task: int,
        num_tasks: int,
        dependencies: list[T.Self] | None = None,
    ):
        if isinstance(job, str):
            self.job = shlex.split(job)
        else:
            self.job = job
        self.cpus_per_task = cpus_per_task
        self.num_tasks = num_tasks
        self.dependencies = dependencies or []
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

        cmd = [
            "srun",
            f"--ntasks={self.num_tasks}",
            f"--cpus-per-task={self.cpus_per_task}",
            "--nodelist=" + ",".join(workers),
            "--exact",
            "--cpu-bind=cores",
            *self.job,
        ]

        LOGGER.debug(" ".join(cmd))
        self.running_on = workers
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


class PyTask(Task):
    pyfunc: T.Callable[[], None]

    def __init__(
        self,
        job: T.Callable[[], None],
        cpus: int = 1,
        dependencies: list[Task] | None = None,
    ):
        super().__init__(
            [sys.executable, "-m", "sim_engine.runpy"], cpus, 1, dependencies
        )
        self.pyfunc = job

    def submit(self, workers: list[str]) -> subprocess.Popen:
        workers = sorted(workers)
        undone_deps = [d for d in self.dependencies if not d.succeeded]
        if undone_deps:
            raise RuntimeError(
                f"Trying to submit task: {self} when the following tasks haven't completed:\n{undone_deps}"
            )

        cmd = [
            "srun",
            f"--ntasks={self.num_tasks}",
            f"--cpus-per-task={self.cpus_per_task}",
            "--nodelist=" + ",".join(workers),
            "--exact",
            "--cpu-bind=cores",
            *self.job,
        ]

        LOGGER.debug(" ".join(cmd))
        self.running_on = workers
        self.child = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
        )
        assert self.child.stdin is not None
        cloudpickle.dump(self.pyfunc, self.child.stdin)
        self.child.stdin.flush()
        self.status = Status.Running
        return self.child
