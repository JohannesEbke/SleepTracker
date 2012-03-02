#!/usr/bin/env python

# -*- coding: utf-8 -*-

import datetime
import sys
import cPickle
import numpy as np
from scipy import signal, interpolate

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess sleeptracker file.')
    parser.add_argument('file')
    args = parser.parse_args()
    if not args.file:
        parser.print_help()
        sys.exit(-1)

    print args.file
    
    t0 = 0 # datetime.datetime(2012, 2, 23, 8, 30, 0)

    t, x, y, z = np.loadtxt(args.file).T

    # get global normalization
    #g = np.sqrt(x**2 + y**2 + z**2)
    #med_g = np.median(g)
    #x/=med_g
    #y/=med_g
    #z/=med_g

    n, timestamps = 0, [t0 + t[0]] # datetime.timedelta(milliseconds=t[0])]
    for i in xrange(1, t.size):
        if t[i] < t[i - 1]:
            n += 1
        #dt = datetime.timedelta(milliseconds=t[i] + n * 2**16)
        dt = t[i] + n * 2**16
        timestamps.append(t0 + dt)

    timestamps = np.array(timestamps)
    t0, t1 = timestamps[0], timestamps[-1]
    new_t = np.linspace(t0, t1, (t1-t0)/25)
    typ = "nearest"
    new_x = interpolate.interp1d(timestamps, x, typ)(new_t)
    new_y = interpolate.interp1d(timestamps, y, typ)(new_t)
    new_z = interpolate.interp1d(timestamps, z, typ)(new_t)

    #results = (timestamps, x, y, z)
    results = (new_t, new_x, new_y, new_z)
    cPickle.dump(results, file(args.file+".pickle", "w"), 2)
    
