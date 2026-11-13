"""
Module to facilitate the execution of python-based tasks.

Reads a pickle-serialized function from stdin and executes it.
If the `--return-task` flag is provided, writes the return value from the function to stdout, serialized with pickle.
"""
import sys

import cloudpickle


def main():
    func = cloudpickle.load(sys.stdin.buffer)
    sys.stdin.close()
    task = func()
    if "--return-task" in sys.argv[1:]:
        cloudpickle.dump(task, sys.stdout.buffer)
        sys.stdout.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
