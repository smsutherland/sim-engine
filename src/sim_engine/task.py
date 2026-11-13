import copy
import logging
import os
import re
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
    """A task to be executed by srun within a slurm job allocation.

    Note that the word "task" gets used to represent a single task within the scheduler,
    as well as individual processes of an MPI-parallelized program.

    Parameters
    ----------
    job: str | list[str]
        If a list, the program to execute and arguments to provide to it
        If a str, the command-line for the task to execute. Will be split using shlex.
    cpus_per_task: int
        How many CPUs to allocate for each MPI task.
    num_tasks: int
        How many MPI tasks to allocate to the job step.
    dependencies: list[Task] | None
        What tasks must run and succeed before this task runs.
    priority: int
        Priority of the task.
        Higher priority tasks will run before lower priority ones, if they are able.
    cwd: str | Path | None
        Directory to run the task inside. If None, task will run in the same working directory as the host Python process.
    modules: list[str] | None
        Lmod modules to load prior to executing the task.
    """

    job: list[str]
    cpus_per_task: int
    num_tasks: int
    dependencies: "list[Task]"
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
        dependencies: "list[Task] | None" = None,
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

    def _update_status(self, sched):
        for d in self.dependencies:
            d._update_status(sched)
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
        """Tell slurm to execute the task on the given nodes.

        Parameters
        ----------
        workers: list[str]
            Names of the nodes to run the task on.

        Returns
        -------
        srun
            Handle to the `srun` process spawned.
        """
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
            "--",
            *self.job,
        ]

        if self.cwd is not None:
            popen_kwargs["cwd"] = self.cwd

        LOGGER.debug(" ".join(cmd))
        self.running_on = workers
        self.child = subprocess.Popen(cmd, **popen_kwargs)
        self.status = Status.Running
        return self.child

    def add_deps(self, new_deps: "Task | list[Task]"):
        if isinstance(new_deps, Task):
            self.dependencies.append(new_deps)
        else:
            self.dependencies.extend(new_deps)
        if self.status == Status.Pending:
            self.status = Status.Waiting

    @property
    def num_cpus(self) -> int:
        """Number of CPUs required for this task."""
        return self.cpus_per_task * self.num_tasks

    @property
    def waiting(self) -> bool:
        """Is the task waiting on its dependencies to run."""
        return self.status == Status.Waiting

    @property
    def pending(self) -> bool:
        """Is the task waiting to be submitted. Dependencies must have finished."""
        return self.status == Status.Pending

    @property
    def running(self) -> bool:
        """Is the task currently running."""
        return self.status == Status.Running

    @property
    def succeeded(self) -> bool:
        """Did the task finish running successfully."""
        return self.status == Status.Succeeded

    @property
    def failed(self) -> bool:
        """Did the task finish running with a non-zero exit code."""
        return self.status == Status.Failed

    @property
    def finished(self) -> bool:
        """Did the task finish running, regardless of success or failure"""
        return self.succeeded or self.failed

    @property
    def dep_failed(self) -> bool:
        """Did one the task's dependencies fail"""
        return self.status == Status.DepFailed

    @property
    def done(self) -> bool:
        """Is the task done, whether because it has already run or because a dependency failed and it will never run."""
        return self.finished or self.dep_failed

    def __str__(self) -> str:
        return f"Task({' '.join(self.job)})"


class PyTask(Task):
    """A python function to be executed as a task within a slurm job allocation.

    Currently MPI parallelized PyTasks are not supported.
    Support is planned.

    Parameters
    ----------
    job: str | list[str]
        If a list, the program to execute and arguments to provide to it
        If a str, the command-line for the task to execute. Will be split using shlex.
    cpus: int
        How many CPUs to allocate for each MPI task.
    dependencies: list[Task] | None
        What tasks must run and succeed before this task runs.
    priority: int
        Priority of the task.
        Higher priority tasks will run before lower priority ones, if they are able.
    cwd: str | Path | None
        Directory to run the task inside. If None, task will run in the same working directory as the host Python process.
    modules: list[str] | None
        Lmod modules to load prior to executing the task.
    """

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
        """Tell slurm to execute the task on the given nodes.

        Parameters
        ----------
        workers: list[str]
            Names of the nodes to run the task on.

        Returns
        -------
        srun
            Handle to the `srun` process spawned.
        """
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
    """A python function to be executed as a task within a slurm job allocation.

    The function should return a new Task (or one of its subclasses) to be executed.
    Any tasks that depend on this TaskGenerator will also depend on the returned task.

    Currently MPI parallelized TaskGenerator are not supported.
    Support is not planned.

    Parameters
    ----------
    job: str | list[str]
        If a list, the program to execute and arguments to provide to it
        If a str, the command-line for the task to execute. Will be split using shlex.
    cpus: int
        How many CPUs to allocate for each MPI task.
    dependencies: list[Task] | None
        What tasks must run and succeed before this task runs.
    priority: int
        Priority of the task.
        Higher priority tasks will run before lower priority ones, if they are able.
    cwd: str | Path | None
        Directory to run the task inside. If None, task will run in the same working directory as the host Python process.
    modules: list[str] | None
        Lmod modules to load prior to executing the task.
    """

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

    def _update_status(self, sched):
        for d in self.dependencies:
            d._update_status(sched)
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
        """Tell slurm to execute the task on the given nodes.

        Parameters
        ----------
        workers: list[str]
            Names of the nodes to run the task on.

        Returns
        -------
        srun
            Handle to the `srun` process spawned.
        """
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


def clone(tasks: list[Task]) -> list[Task]:
    """Creates a copy of the given list of tasks, keeping task dependencies isolated from the original copy.

    Will raise an exception if any task provided depends on a task not in the list.

    Parameters
    ----------
    tasks: list[Task]
        Tasks to clone.

    Returns
    -------
    tasks: list[Task]
        A copy of all the tasks.
    """
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
    """Create multiple clones of the provided task list, each executing in one of the provided paths.

    If a task already has a cwd set, then the cwd will be changed to be relative to each path.
    If the cwd is an absolute path, then it will not be modified.

    Parameters
    ----------
    tasks: list[Task]
        Tasks to multiply.
    paths: list[Path]
        Paths to run the set of tasks within.

    Returns
    -------
    tasks: list[Task]
    """
    new_tasks = []
    for p in paths:
        tasks_copy = clone(tasks)
        for t in tasks_copy:
            if t.cwd is not None:
                t.cwd = p / t.cwd
            else:
                t.cwd = p
        new_tasks.extend(tasks_copy)
    return new_tasks


def series(tasks: list[Task]) -> list[Task]:
    """Make each task in a list depend on the previous.

    Modifications happen in-place and a reference to the task list is returned for convenience.

    Parameters
    ----------
    tasks: list[Task]
        Series of tasks to make depend on each other.

    Returns
    -------
    tasks: list[Task]
        Identical to the supplied task list.
    """
    for i, t in enumerate(tasks[1:]):
        t.dependencies.append(tasks[i])
    return tasks
