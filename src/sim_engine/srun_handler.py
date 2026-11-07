import os
import sys


def main():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    MPI.Finalize()
    child_envirion = os.environ.copy()
    cpu_mask = os.environ.get("SLURM_CPU_MASK_LIST")
    if cpu_mask is not None:
        cpu_mask = int(cpu_mask, 16)
        mask_str = ""
        while cpu_mask:
            if cpu_mask & 1:
                mask_str
            cpu_mask >>= 1

    if rank == 0:
        os.execvpe(sys.argv[1], sys.argv[1:], child_envirion)


if __name__ == "__main__":
    sys.exit(main())
