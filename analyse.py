# -*- coding: utf-8 -*-

import datetime

from matplotlib import pyplot
import numpy as np

fname = "RUN2.TXT"
t0 = datetime.datetime(2012, 2, 23, 8, 30, 0)

t, x, y, z = np.loadtxt(fname).T

n, timestamps = 0, [t0 + datetime.timedelta(milliseconds=t[0])]
for i in xrange(1, t.size):
    if t[i] < t[i - 1]:
        n += 1
    dt = datetime.timedelta(milliseconds=t[i] + n * 2**16)
    timestamps.append(t0 + dt)

timestamps = np.array(timestamps)

figure = pyplot.figure(figsize=(10, 3))

pyplot.plot(timestamps, sqrt(x**2 + y**2 + z**2), label='X')
#pyplot.plot(timestamps, x, label='X')
#pyplot.plot(timestamps, y, label='Y')
#pyplot.plot(timestamps, z, label='Z')

pyplot.xlim(t0)

pyplot.xlabel("Time")
pyplot.ylabel("ADC value")

pyplot.legend()

pyplot.savefig(fname[:-4] + '.png', dpi=150, bbox_inches='tight')
pyplot.show()
pyplot.close(figure)
