from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
from tqdm import tqdm
import sys
import os


def parallel(func, x, message=None, *args, **kwargs):
    '''
    Run some function in parallel.

    inputs:
        func = The function to run.
        x = The list of items to iterate on.
        message = A message to print.
        args = Arguemnts for func.
        kwargs = Keyword arguments for func.

    outputs:
        data = A list of data for each item, x.
    '''

    if message:
        print(message)

    part_func = partial(func, *args, **kwargs)
    with Pool(os.cpu_count()) as pool:
        data = list(tqdm(
                         pool.imap(part_func, x),
                         total=len(x),
                         file=sys.stdout
                         ))

    return data
