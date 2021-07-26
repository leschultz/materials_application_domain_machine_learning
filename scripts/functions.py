from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

import sys
import os


def parallel(func, x, *args, **kwargs):
    '''
    Run some function in parallel.
    '''

    pool = Pool(os.cpu_count())
    part_func = partial(func, *args, **kwargs)

    data = list(tqdm(pool.imap(part_func, x), total=len(x), file=sys.stdout))
    pool.close()
    pool.join()

    return data
