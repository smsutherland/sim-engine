import sys

import cloudpickle


def main():
    func = cloudpickle.load(sys.stdin.buffer)
    task = func()
    cloudpickle.dump(task, sys.stdout.buffer)
    sys.stdout.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
