import re
import copy
import os
import logging
import shlex
import subprocess
import sys
import typing as T
from enum import Enum
from pathlib import Path

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
    cwd: Path | None = None
    modules: list[str]

    def __init__(
        self,
        job: str | list[str],
        cpus_per_task: int,
        num_tasks: int,
        dependencies: list[T.Self] | None = None,
        priority: int = 0,
        cwd: str | Path | None = None,
        modules: list[str] | None = None,
    ):
        if isinstance(job, str):
            self.job = shlex.split(job)
        else:
            self.job = job
        self.cpus_per_task = cpus_per_task
        self.num_tasks = num_tasks
        self.dependencies = dependencies or []
        self.priority = priority
        self.modules = modules or []
        if cwd is None:
            self.cwd = None
        else:
            self.cwd = Path(cwd)
        if self.dependencies:
            self.status = Status.Waiting
        else:
            self.status = Status.Pending

    def update_status(self, sched):
        for d in self.dependencies:
            d.update_status(sched)
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
        return self._submit(workers, {})

    def _submit(
        self, workers: list[str], popen_kwargs: dict[str, T.Any]
    ) -> subprocess.Popen:
        workers = sorted(workers)
        undone_deps = [d for d in self.dependencies if not d.succeeded]
        if undone_deps:
            raise RuntimeError(
                f"Trying to submit task: {self} when the following tasks haven't completed:\n{undone_deps}"
            )

        if self.modules:
            if (lmod := os.environ.get("LMOD_CMD", None)) is None:
                raise RuntimeError("Trying to use modules with no $LMOD_CMD!")
            result = subprocess.check_output(
                [lmod, "bash", "load", *self.modules],
                encoding="utf8",
                stderr=subprocess.DEVNULL,
            )
            child_environ = (
                popen_kwargs["env"].copy()
                if "env" in popen_kwargs
                else os.environ.copy()
            )
            vars = os.environ.copy()
            for line in result.splitlines():
                if match := re.match(r"(\w+)=(.*);", line):
                    vars[match[1]] = match[2]
                elif match := re.match(r"export (\w+);", line):
                    child_environ[match[1]] = vars[match[1]]
            popen_kwargs["env"] = child_environ

        cmd = [
            "srun",
            f"--ntasks={self.num_tasks}",
            f"--cpus-per-task={self.cpus_per_task}",
            "--nodelist=" + ",".join(workers),
            "--exact",
            "--cpu-bind=cores",
            *self.job,
        ]

        if self.cwd is not None:
            popen_kwargs["cwd"] = self.cwd

        LOGGER.debug(" ".join(cmd))
        self.running_on = workers
        self.child = subprocess.Popen(cmd, **popen_kwargs)
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
        priority: int = 0,
        cwd: str | Path | None = None,
        modules: list[str] | None = None,
    ):
        super().__init__(
            [sys.executable, "-m", "sim_engine.runpy"],
            cpus,
            1,
            dependencies,
            priority,
            cwd,
            modules,
        )
        self.pyfunc = job

    def submit(self, workers: list[str]) -> subprocess.Popen:
        child = self._submit(
            workers,
            {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
            },
        )
        assert child.stdin is not None
        cloudpickle.dump(self.pyfunc, child.stdin)
        child.stdin.flush()
        return child


class TaskGenerator(Task):
    pyfunc: T.Callable[[], Task]
    buffered_stdout: bytes = b""

    def __init__(
        self,
        job: T.Callable[[], Task],
        cpus: int = 1,
        dependencies: list[Task] | None = None,
        priority: int = 0,
        cwd: str | Path | None = None,
        modules: list[str] | None = None,
    ):
        super().__init__(
            [sys.executable, "-m", "sim_engine.runpy", "--return-task"],
            cpus,
            1,
            dependencies,
            priority,
            cwd,
            modules,
        )
        self.pyfunc = job

    def update_status(self, sched):
        for d in self.dependencies:
            d.update_status(sched)
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
                assert self.child.stdout is not None
                self.buffered_stdout += self.child.stdout.read(1024)
                if (retval := self.child.poll()) is not None:
                    if retval == 0:
                        self.buffered_stdout += self.child.stdout.read()
                        new_task = cloudpickle.loads(self.buffered_stdout)
                        for t in sched.tasks:
                            if self in t.dependencies:
                                t.dependencies.append(new_task)
                        sched.tasks.append(new_task)
                        self.status = Status.Succeeded
                    else:
                        self.status = Status.Failed

    def submit(self, workers: list[str]) -> subprocess.Popen:
        child = self._submit(
            workers,
            {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
            },
        )
        assert child.stdin is not None
        cloudpickle.dump(self.pyfunc, child.stdin)
        child.stdin.flush()
        return child


def clone_tasks(tasks: list[Task]) -> list[Task]:
    dep_indices = []
    for t in tasks:
        di = []
        for d in t.dependencies:
            di.append(tasks.index(d))
        dep_indices.append(di)

    new_tasks = [copy.copy(t) for t in tasks]
    for t, di in zip(new_tasks, dep_indices):
        t.dependencies = [new_tasks[i] for i in di]
    return new_tasks


def multiply(tasks: list[Task], paths: list[Path]) -> list[Task]:
    new_tasks = []
    for p in paths:
        tasks_copy = clone_tasks(tasks)
        for t in tasks_copy:
            if t.cwd is not None:
                t.cwd = p / t.cwd
            else:
                t.cwd = p
        new_tasks.extend(tasks_copy)
    return new_tasks
