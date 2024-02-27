from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
from tqdm import tqdm
import sys
import os


def parallel(
             func,
             x,
             message=None,
             disable=False,
             n_jobs=-1,
             *args,
             **kwargs,
             ):
    '''
    Run some function in parallel.

    inputs:
        func = The function to run.
        x = The list of items to iterate on.
        message = A message to print.
        disable = Disable tqdm print.
        n_jobs = The number of cores to run on.
        args = Arguemnts for func.
        kwargs = Keyword arguments for func.

    outputs:
        data = A list of data for each item, x.
    '''

    if message:
        print(message)

    part_func = partial(func, *args, **kwargs)

    if n_jobs == -1:
        n_jobs = os.cpu_count()
    else:
        n_jobs = n_jobs

    with Pool(n_jobs) as pool:
        data = list(tqdm(
                         pool.imap(part_func, x),
                         total=len(x),
                         file=sys.stdout,
                         disable=disable,
                         ))

    return data
