#!/usr/bin/env pythonfx
#
# This file is part of FIREwork
# 
# FIREwork is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIREwork is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
#
# FIREwork Copyright (C) 2008 - 2011 Julien Bert 

# ==== Utils ================================
# ===========================================

# barrier function
def wait():
    raw_input('WAITING [Enter]')
    return

# convert engineer prefix
def prefix_SI(mem):
    from math import log
    
    pref   = ['', 'k', 'M', 'G', 'T']
    iemem  = int(log(mem) // log(1e3))
    mem   /= (1e3 ** iemem)

    return '%5.2f %s' % (mem, pref[iemem])

# convert time format to nice format
def time_format(t):
    time_r  = int(t)
    time_h  = time_r // 3600
    time_m  = (time_r % 3600) // 60
    time_s  = (time_r % 3600)  % 60
    time_ms = int((t - time_r) * 1000.0)
    txt     = ' %03i ms' % time_ms
    if time_s != 0: txt = ' %02i s' % time_s + txt
    if time_m != 0: txt = ' %02i m' % time_m + txt
    if time_h != 0: txt = ' %02i h' % time_h + txt
    return txt

# plot
def plot(x, y):
    import matplotlib.pyplot as plt
    
    plt.plot(x, y)
    plt.show()

# plot points distribution
def plot_dist(x, y):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    nullfmt   = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8,8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)
    
    # now determine nice limits by hand:
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    
    binwidth = (xmax - xmin) / 100.0
    binheight = (ymax - ymin) / 100.0
    
    #xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    #lim = ( int(xymax/binwidth) + 1) * binwidth

    #axScatter.set_xlim( (-lim, lim) )
    axScatter.set_xlim((xmin, xmax))
    #axScatter.set_ylim( (-lim, lim) )
    axScatter.set_ylim(ymin, ymax)
    
    binsw = np.arange(xmin, xmax + binwidth, binwidth)
    binsh  = np.arange(ymin, ymax + binheight, binheight)
    axHistx.hist(x, bins=binsw)
    axHisty.hist(y, bins=binsh, orientation='horizontal')

    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )

    plt.show()
    
# plot RAPS curve
def plot_raps(im):
    import matplotlib.pyplot as plt
    from numpy    import log
    from firework import image_raps
    
    #im = image_normalize(im)
    val, freq = image_raps(im)
    #val = image_atodB(val)
    #val = log(val)
    plt.semilogy(freq, val)
    plt.xlabel('Nyquist frequency')
    plt.ylabel('Power spectrum')
    plt.grid(True)
    plt.show()

# plot FRC curve
def plot_frc(im1, im2):
    import matplotlib.pyplot as plt
    from firework import image_frc
    
    frc, freq = image_frc(im1, im2)
    plt.plot(freq, frc)
    plt.xlabel('Nyquist frequency')
    plt.ylabel('FRC')
    plt.axis([0, 0.5, 0, 1.0])
    plt.grid(True)
    plt.show()

# plot profil of any filter
def plot_filter_profil(H):
    import matplotlib.pyplot as plt
    from firework import filter_profil
    
    p, f = filter_profil(H)
    plt.plot(f, p)
    plt.axhline(y=0.707, c='r', ls=':')
    plt.axhline(y=0.5, c='r', ls=':')
    plt.xlabel('Nyquist frequency')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.axis([0, 0.5, 0, p.max()])
    plt.show()
    
# smooth curve
def curve_smooth(a, order):
    from numpy import zeros

    for o in xrange(order):
        b     = zeros(a.shape, a.dtype)
        n     = a.size - 1
        b[-1] = a[-1]
        for i in xrange(n):
            b[i] = (a[i] + a[i+1]) / 2.0

        a = b.copy()

    return a

# return 1D hitogram based on 1D data
def vector_hist(data, nbins):
    import matplotlib.pyplot as plt
    import matplotlib.mlab   as mlab

    n, bins, patches = plt.hist(data, nbins, facecolor='green', alpha=0.75)
    return bins, n

