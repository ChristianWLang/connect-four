"""
Main algorithm.
"""
# Author: Christian Lang <me@christianlang.io>

from .utils.random_game import random_game

import datetime as dt
from tqdm import tqdm


def main():
    now = dt.datetime.now()
    for i in tqdm(range(10000)):
        record, winner = random_game()
    end = dt.datetime.now()
    print(end - now)
    return
if __name__ == '__main__':
    main()
