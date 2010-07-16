#!/usr/bin/env python
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
# FIREwork Copyright (C) 2008 - 2010 Julien Bert 

def image_show(mat):
    import matplotlib.pyplot as plt
    import os, sys

    h, w = mat.shape
    fig  = plt.figure()
    ax   = fig.add_subplot(111)
    cax  = ax.imshow(mat, interpolation='nearest', cmap='hot')
    ax.set_title('Viewer - FIREwork : %i x %i' % (w, h))
    min  = mat.min()
    max  = mat.max()
    d    = (max - min) / 9.0
    lti  = [i*d+min for i in xrange(10)]
    txt  = ['%5.3f' % lti[i] for i in xrange(10)]
    cbar = fig.colorbar(cax, ticks=lti)
    cbar.ax.set_yticklabels(txt)
    plt.show()
    
# plot 1D hitogram based on 1D data
def hist1D_plot(data, nbins):
    import matplotlib.pyplot as plt
    import matplotlib.mlab   as mlab

    n, bins, patches = plt.hist(data, nbins, facecolor='green', alpha=0.75)
    #print n
    #print bins
    #plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    plt.title('Viewer - FIREwork hist1D')
    #plt.axis([min(data), max(data), 0, max(n)])
    plt.grid(True)
    
    plt.show()
