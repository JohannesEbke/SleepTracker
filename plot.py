#!/usr/bin/env python

# -*- coding: utf-8 -*-

import datetime
import numpy as np
import scipy as sp
from scipy import fftpack as fftp
import sys
import cPickle
from matplotlib import pyplot, ticker, cm
from scipy import signal, interpolate
from math import sin, cos, acos, sqrt, floor

avg_ms = 44
avg_per_sec = 1000.0/avg_ms
avg_per_min = 60 * avg_per_sec

def hsvToRGB(h, s, v):
    """Convert HSV color space to RGB color space

    @param h: Hue
    @param s: Saturation
    @param v: Value
    return (r, g, b)  
    """
    hi = floor(h / 60.0) % 6
    f =  (h / 60.0) - floor(h / 60.0)
    p = v * (1.0 - s)
    q = v * (1.0 - (f*s))
    t = v * (1.0 - ((1.0 - f) * s))
    return {
        0: [v, t, p],
        1: [q, v, p],
        2: [p, v, t],
        3: [p, q, v],
        4: [t, p, v],
        5: [v, p, q],
    }[hi]

def norm_angle(x):
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x

def complex_to_rgb(x):
    angle = norm_angle(np.angle(x))

    h = norm_angle(angle)/2/np.pi
    s = cos(angle/4)
    v = np.abs(x)/1000.0
    v = min(1,v)
    assert 0 <= h <= 1
    assert 0 <= s <= 1
    assert 0 <= v <= 1
    return hsvToRGB(255*h, s, v)

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

def get_spectrogram(x, width, skip):
    lines = []
    for i in xrange((len(x) - width)/skip):
        #print i, "/", (len(x) - width)/skip
        ft = fftp.rfft(x[i*skip:i*skip+width])[1:]
        lines.append(abs(ft))
        #lines.append(ft**2)
        #lines.append(np.array(map(complex_to_rgb, ft)))
    res = np.array(lines)
    res = np.swapaxes(res, 0, 1)
    return res

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

    if True:
        print len(t)
        #start, end = (0,-400000)
        #start, end = (40*60*5, 40*(3600))
        start, end = (40*60*5, -1)
        t = t[start:end]
        x = x[start:end]
        y = y[start:end]
        z = z[start:end]

    if args.window != 1:
        x = get_moving_average(x, args.window)
        y = get_moving_average(y, args.window)
        z = get_moving_average(z, args.window)

    assert len(t) == len(x)

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dz = z[1:] - z[:-1]
    #t = t[1:]
    g = np.sqrt(x**2 + y**2 + z**2)

    if False:
        dg = np.sqrt(dx**2 + dy**2 + dz**2)    
        #figure = pyplot.figure(figsize=(10, 3))

        window1s, windows10s = int(avg_per_sec), 10*int(avg_per_sec)

        def plot_avg_win(window):
            avgval, rmss = get_moving_average(dg, window), get_moving_rms(dg, window)
            #ang = get_angular_change(get_moving_average(dx, window), get_moving_average(dy, window), get_moving_average(dz, window))
            print len(t), len(avgval)


            #pyplot.plot(t[1:], avgval, label='G [%i]' % window)
            #pyplot.plot(t[1:], rmss, label='G [%i] RMS' % window)
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


    print "Sigma Counts: "
    xv = sum(trig.counter.values())
    cum = 0
    for sigma in sorted(trig.counter):
        cum += trig.counter[sigma]
        print sigma+1, trig.counter[sigma], cum*100.0/xv

    #pyplot.plot(t[1:], dg, label='Delta G')
    #g_fft = sp.fft(dg)
    #pyplot.plot(t[1:], g_fft, label='G (FFT)')
    #med_g = signal.medfilt(g, 1001)
    #pyplot.plot(t, med_g, label='G (Median)')
    #pyplot.plot(t, g, label='G')
    #pyplot.plot(t, x, label='X')
    #pyplot.plot(t, y, label='Y')
    #pyplot.plot(t, z, label='Z')

    #pyplot.xlim(t[0])

    #pyplot.xlabel("Time")
    #pyplot.ylabel("ADC value")

    #pyplot.legend()

    #pyplot.savefig(args.file + '.png', dpi=150, bbox_inches='tight')
    #pyplot.show()
    #pyplot.close(figure)


    fft_width = 40*5 # 10 sec should be enough to catch all regular human rythms
    fft_skip = fft_width # only one per second
    sg = get_spectrogram(g, fft_width, fft_skip)
    #print sg
    def xv_to_time(x, pos):
        return "%.2f" % (x*fft_skip/40.0)
    def yv_to_freq(y, pos):
        return "%.1f Hz" % (y/(fft_width/40.0))

    figure = pyplot.figure(figsize=(10, 3))
    sfig1 = figure.add_subplot(211)
    sfig1.imshow(sg, aspect="auto", cmap=cm.hot)
    sfig1.set_ylabel("Frequency")
    sfig1.set_xlabel("Time")
    sfig1.get_xaxis().set_major_formatter(ticker.FuncFormatter(xv_to_time))
    sfig1.get_yaxis().set_major_formatter(ticker.FuncFormatter(yv_to_freq))


    sfig2 = figure.add_subplot(212, sharex=sfig1)
    timeax = np.linspace(0,len(sg[0]), len(ts))
    sfig2.plot(timeax, ts, label='Trigger Sigma')
    #sfig2.plot(timeax, td, label='Trigger Moving Mean Square')
    sfig2.plot(timeax, trigs, label='Triggers')
    sfig2.get_xaxis().set_major_formatter(ticker.FuncFormatter(xv_to_time))



    figure.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.99, wspace=0.0, hspace=0.0)
    figure.savefig(args.file + '.png', dpi=150, bbox_inches='tight')
    pyplot.show()
    pyplot.close(figure)
