#!/usr/bin/env python

import os
import torch
import torch.distributed as dist
import time
import json

import argparse

from multiprocessing import Process, Queue, set_start_method

parser = argparse.ArgumentParser()
parser.add_argument("world_size", type=int)
parser.add_argument("--host", default='localhost')
parser.add_argument("--port", type=int, default=4992)
parser.add_argument("--mb", type=float, default=24.0)
parser.add_argument('--N', type=int, default=97)
parser.add_argument('--backend', default='nccl')
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--algo', default=None)

def f(args, rank, queue):
    if args.verbose:
        os.environ['NCCL_DEBUG'] = 'INFO'
    if args.algo:
        os.environ['NCCL_ALGO'] = args.algo
    dist.init_process_group(backend=args.backend, init_method=f"tcp://{args.host}:{args.port}", world_size=args.world_size, rank=rank)
    device_id = f"cuda:{rank}"
    mb, N = args.mb,args.N
    torch.cuda.set_device(device_id)
    small = torch.empty(1, device=device_id)
    buffer = torch.empty(int(mb * 1024 * 1024 / 2), device=device_id, dtype=torch.float16)

    dist.all_reduce(small) # Warmming up

    times = []
    for i in range(N):
        torch.cuda.synchronize(device_id)
        st = time.time()
        dist.all_reduce(buffer)
        torch.cuda.synchronize(device_id)
        torch.cuda.synchronize(device_id)
        ed = time.time()
        times.append(ed - st)

        time.sleep(0.2)
    queue.put((rank, times))
    dist.destroy_process_group()

if __name__ == '__main__':
    set_start_method('spawn')
    queue = Queue()
    args = parser.parse_args()
    world_size = args.world_size
    procs = [Process(target=f, args=(args, rank, queue)) for rank in range(world_size)]
    for proc in procs: proc.start()

    results = []
    for _ in range(world_size):
        results.append(queue.get())
    for proc in procs:
        proc.join()
    print(json.dumps(dict(results)))
