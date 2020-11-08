import os
import time


def logging(file):
    def write_log(s):
        print(s)
        with open(file, 'a') as f:
            f.write(s+'\n')

    return write_log

