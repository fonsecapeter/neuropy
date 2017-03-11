# -*- coding: utf-8 -*-
from __future__ import print_function

# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=100,
                   fill='█', blank='-', left_cap='|', right_cap='|', show_percent=True):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + blank * (length - filledLength)
    if show_percent:
        print('\r%s %s%s%s %s%% %s' % (prefix, left_cap, bar, right_cap, percent, suffix), end='\r')
    else:
        print('\r%s %s%s%s' % (prefix, left_cap, bar, right_cap), end='\r')

    # Print New Line on Complete
    if iteration >= total:
        print()


"""
#
# Sample Usage
#

from time import sleep

# make a list
items = list(range(0, 57))
i = 0
l = len(items)

# Initial call to print 0% progress
printProgressBar(i, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
for item in items:
    # Do stuff...
    sleep(0.1)
    # Update Progress Bar
    i += 1
    printProgressBar(i, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

# Sample Output
Progress: |█████████████████████████████████████████████-----| 90.0% Complete
"""
