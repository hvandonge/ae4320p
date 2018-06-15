# -*- coding: utf-8 -*-
"""
Created on Thu May 31 10:36:41 2018

@author: Henk
"""

import sys
toolbar_width = 40

def bar_init(toolbar_width=40):
    """Function that initializes the progress bar"""
    # setup toolbar
    sys.stdout.write('['+'{}'.format(' ' * toolbar_width)+']'+' {:4.0%}'.format(0))
    sys.stdout.flush()

def bar(i, count):
    """Function that updates the progress bar"""
    j = float(i+1)
    sys.stdout.write('\r')
    sys.stdout.write('['+'{}'.format('#'*int(round(j/count*toolbar_width))) +
                       '{}'.format(' '*int(round((count-j)/count*toolbar_width))) +
                       ']'+' {:4.0%}'.format(j/count))
    sys.stdout.flush()

#    sys.stdout.write('\r[')
#    sys.stdout.write('#'*int(float(i+1)/count*toolbar_width))
#    sys.stdout.flush()