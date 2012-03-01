#!/usr/bin/env python

# -*- coding: utf-8 -*-

import datetime
import numpy as np
import scipy as sp
import sys
import cPickle
from matplotlib import pyplot
from scipy import signal
from math import acos, sqrt

avg_ms = 44
avg_per_sec = 1000.0/avg_ms
avg_per_min = 60 * avg_per_sec

def get_moving_average(data, iwindow):
    res = sum(data[i:-iwindow+i] for i in range(iwindow))
    res /= iwindow
    return np.array([0.0]*iwindow + [x for x in res])

def get_moving_rms(data, iwindow):
    mean = sum(data[i:-iwindow+i] for i in range(iwindow))/iwindow
    res = sum((data[i:-iwindow+i]-mean)**2 for i in range(iwindow))
    res /= iwindow
    return [0.0]*iwindow + [x for x in np.sqrt(res)]

def get_angular_change(x, y, z):
    res = [0.0]
    vabs = [x[i]*x[i] + y[i]*y[i] + z[i]*z[i] for i in xrange(len(x))]
    return [0.0] + [acos((x[i]*x[i-1] + y[i]*y[i-1] + z[i]*z[i-1])/(vabs[i]*vabs[i-1]+0.00001)) for i in xrange(1,len(x))]


class Trigger():
    def __init__(self):
        self.moving_mean_square = 0.0
        self.rms_exp = 1.0/(10*avg_per_sec)
        self.prev = (0.0, 0.0, 0.0)
        self.sigma = 0
        self.counter = {}
        self.dv = 0.0

    def update(self, v):
        self.moving_mean_square *= (1.0 - self.rms_exp)
        self.moving_mean_square += self.rms_exp * (v**2)

    def tick(self, x, y, z):
        px, py, pz = self.prev
        dx, dy, dz = (x-px, y-py, z-pz)
        self.prev = (x, y, z)
        v = sqrt(dx**2 + dy**2 + dz**2)
        if self.moving_mean_square > 0:
            self.sigma = v/sqrt(self.moving_mean_square)
            self.counter[int(self.sigma)] = self.counter.get(int(self.sigma), 0) + 1
            self.dv = v - 3*sqrt(self.moving_mean_square)

        #if self.sigma < 5:
        self.update(v)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot sleeptracker file.')
    parser.add_argument('file')
    parser.add_argument('-w', '--window', metavar='N', default=1, type=int, help='average over these many samples')
    args = parser.parse_args()
    if not args.file:
        parser.print_help()
        sys.exit(-1)

    # time is in milliseconds
    t, x, y, z = cPickle.load(file(args.file))

    if args.window != 1:
        x = get_moving_average(x, args.window)
        y = get_moving_average(y, args.window)
        z = get_moving_average(z, args.window)

    if True:
        print len(t)
        start, end = (0,-400000)
        t = t[start:end]
        x = x[start:end]
        y = y[start:end]
        z = z[start:end]

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dz = z[1:] - z[:-1]
    #t = t[1:]
    #g = np.sqrt(x**2 + y**2 + z**2)
    dg = np.sqrt(dx**2 + dy**2 + dz**2)

    figure = pyplot.figure(figsize=(10, 3))

    window1s, windows10s = int(avg_per_sec), 10*int(avg_per_sec)

    def plot_avg_win(window):
        avgval, rmss = get_moving_average(dg, window), get_moving_rms(dg, window)
        #ang = get_angular_change(get_moving_average(dx, window), get_moving_average(dy, window), get_moving_average(dz, window))
        pyplot.plot(t, avgval, label='G [%i]' % window)
        pyplot.plot(t, rmss, label='G [%i] RMS' % window)
        #pyplot.plot(t, ang1s, label='Angle change [1s]')

    #plot_avg_win(1)
    #plot_avg_win(1 *int(avg_per_sec))
    #plot_avg_win(10*int(avg_per_sec))
    trig = Trigger()
    ts, td, trigs = [], [], []
    for i in xrange(len(t)):
        trig.tick(x[i],y[i],z[i])
        ts.append(trig.sigma)
        td.append(trig.moving_mean_square/500.0)
        trigs.append(trig.dv if trig.sigma > 3 else 0)

    pyplot.plot(t, ts, label='Trigger Sigma')
    pyplot.plot(t, td, label='Trigger Moving Mean Square')
    pyplot.plot(t, trigs, label='Triggers')

    print "Sigma Counts: "
    xv = sum(trig.counter.values())
    cum = 0
    for x in sorted(trig.counter):
        cum += trig.counter[x]
        print x+1, trig.counter[x], cum*100.0/xv

    pyplot.plot(t, [0.0] + [x/100.0 for x in dg], label='Delta G')
    #g_fft = sp.fft(g)
    #pyplot.plot(t, g_fft, label='G (FFT)')
    #med_g = signal.medfilt(g, 1001)
    #pyplot.plot(t, med_g, label='G (Median)')
    #pyplot.plot(t, g, label='G')
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
