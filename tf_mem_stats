#!/usr/bin/env python

import re
import sys
import os
import subprocess
from collections import Counter, defaultdict

allocation_re = re.compile(br'At time (\d+) (de)?allocated (\d+) for tensor'
                           br' (.+)')


def main(argv):
    proc = start_proc(argv[1:])
    stats = debug_process(proc)
    print_stats(*stats)
    proc.wait()


def start_proc(args):
    environ = os.environ.copy()
    environ['PYTHONUNBUFFERED'] = '1'
    environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

    return subprocess.Popen(args, env=environ, stderr=subprocess.PIPE)


def debug_process(proc):
    allocations = Counter()
    max_allocations = defaultdict(int)
    total_allocation = 0
    max_total_allocation = 0
    max_allocations_snapshot = allocations.copy()
    max_time = 0
    for line in proc.stderr:
        match = allocation_re.search(line)
        if not match:
            continue

        time, deallocation, size, tensor = match.groups()
        size = int(size)
        tensor = tensor.decode()
        time = int(time)
        if deallocation:
            size = -size
        allocations[tensor] += size
        max_allocations[tensor] = max(
            allocations[tensor], max_allocations[tensor]
        )

        total_allocation += size
        if max_total_allocation < total_allocation:
            max_total_allocation = total_allocation
            max_allocations_snapshot = allocations.copy()
            max_time = time

    return list((v, k) for k, v in max_allocations_snapshot.items()), \
           max_total_allocation, max_time


def print_stats(allocations, max_total_allocation, max_time):
    print(f'max {max_total_allocation} at {max_time}')

    top = sorted(allocations, reverse=True)[:100]
    for size, tensor in top:
        print(f'{size} - {tensor}')


if __name__ == '__main__':
    main(sys.argv)
