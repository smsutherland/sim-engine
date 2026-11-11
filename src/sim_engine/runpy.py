import sys
import time

import cloudpickle


def main():
    time.sleep(3)
    func = cloudpickle.load(sys.stdin.buffer)
    sys.stdin.close()
    task = func()
    if "--return-task" in sys.argv[1:]:
        cloudpickle.dump(task, sys.stdout.buffer)
        sys.stdout.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
