# sim-engine
`sim-engine` is a slurm-based task scheduler optimized for running and analyzing
large astrophysical simulations.

At its most basic, you create some tasks and have them executed within a slurm
job allocation.

```py
import sim_engine as se

tasks = [
    ic_task := se.Task(
        "2LPTic 2LPT.param",
        1, 8,
        modules=["openmpi", "hdf5", "fftw/mpi-2", "gsl"],
    ),
    se.Task(
        "swift-mpi --threads=32 --pin params.yml",
        32, 8, [ic_task],
        modules=["openmpi", "hdf5", "fftw/mpi-3", "gsl", "jemalloc"]
    ),
    # And perhaps some post-processing you'd like to perform on the simulation outputs
]
sched = se.Scheduler(tasks)
sched.run()
```

# Features
### Task dependencies
Tasks can depend on other tasks. A task will only run when its dependencies have
successfully run. If a dependency fails, then the task will not run.
`sim_engine.series` can be used to simply create a series of tasks that run in
order.

### Task priority
If a task with high priority can run (its dependencies have succeeded), then the
scheduler will wait for space enough to be available and run the high-priority
tasks. This prevents small low-priority tasks from starving larger tasks if they
have a higher priority.
