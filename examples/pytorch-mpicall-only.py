#!/usr/bin/env python
import os, sys
import socket
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run(mpi_params, hostname):
    size, rank = mpi_params[0], mpi_params[2]
    group = dist.new_group([i for i in range(size)])
    tensor = torch.zeros(size)
    tensor[rank] = 1

    print(f"I am {rank} of {size} in {hostname}", "world_size={0}, local_size={1}, rank={2}, local_rank={3}, node_rank={4}".format(*mpi_params))
    print(f'Rank {rank}', ' initial data ', tensor)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    #dist.reduce(tensor, 0, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor)

def init_processes(mpi_params, hostname, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    size, local_size, rank, local_rank, node_rank = mpi_params
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(mpi_params, hostname)


if __name__ == "__main__":
    if 'OMPI_COMM_WORLD_SIZE' not in os.environ:
        print ("Test how pytorch can broadcast tensors via MPI call between <N> processes")
        print ("example to run 100 MPI procs: mpirun -np 100 python pytorch-mpicall-only.py 100")
        sys.exit()

    mpi_params = [int(os.environ[x]) for x in ["OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", \
                                   "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "OMPI_COMM_WORLD_NODE_RANK"]]
    hostname = socket.gethostname()
    print (hostname, mpi_params)
    init_processes(mpi_params, hostname, run, backend='mpi')
