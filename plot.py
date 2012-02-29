#!/usr/bin/env python

# -*- coding: utf-8 -*-

import datetime
import numpy as np
import scipy as sp
import sys
import cPickle
from matplotlib import pyplot
from scipy import signal

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot sleeptracker file.')
    parser.add_argument('file')
    args = parser.parse_args()
    if not args.file:
        parser.print_help()
        sys.exit(-1)

    # time is in milliseconds
    t, x, y, z = cPickle.load(file(args.file))

    if False:
        start, end = (3000,4000)
        t = t[start:end]
        x = x[start:end]
        y = y[start:end]
        z = z[start:end]

    g = np.sqrt(x**2 + y**2 + z**2)
    #g_fft = sp.fft(g)
    #med_g = signal.medfilt(g, 1001)
    #avg_g = signal.medfilt(g, 1001)

    figure = pyplot.figure(figsize=(10, 3))

    #pyplot.plot(t, g_fft, label='G (FFT)')
    #pyplot.plot(t, med_g, label='Med G')
    #pyplot.plot(t, g_fft, label='G')
    pyplot.plot(t, g, label='G')
    #pyplot.plot(t, x, label='X')
    #pyplot.plot(t, y, label='Y')
    #pyplot.plot(t, z, label='Z')

    pyplot.xlim(t[0])

    pyplot.xlabel("Time")
    pyplot.ylabel("ADC value")

    pyplot.legend()

    pyplot.savefig(args.file + '.png', dpi=150, bbox_inches='tight')
    pyplot.show()
    pyplot.close(figure)
