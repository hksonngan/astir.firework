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

import optparse, os, sys

progname = os.path.basename(sys.argv[0])
usage    = progname + ' source_name.bin target_name'
topic    = 'Convert LOR binary data (ID crystals and modules) to position in space according PET geometry'
p        = optparse.OptionParser(usage, description=topic)
p.add_option('--convert', action='store', type='choice', choices=['int', 'float'], default='int', help='["int" or "float"] Convert in integer for DDA projector or in float point to Siddon projector (default int)')
p.add_option('--Nstep',    type='int',    default=1,       help='Split the job in Nstep iteration to avoid overloading memory (default 1)')
p.add_option('--rnd',    type='int',    default=10,       help='Seed random number to determine random position on the crystal (default 10). If set to 0 no random process')

(options, args) = p.parse_args()
if len(args) < 2:
    print topic
    print ''
    print 'usage:', usage
    print ''
    print 'please run "' + progname + ' -h" for detailed options'
    sys.exit()

src    = args[0]
trg    = args[1]
kind   = options.convert
Nstep  = options.Nstep
rnd    = options.rnd

from firework import *
from numpy    import *
from time     import time

sizexy_space = 1004 # xy scanner inside im of 251 pix * respix
sizez_space  = 180  # z scanner inside im of 45 pix * respix
respix       = 565.0/141.0
#respix = 4.0
sizexy_im    = 141  # gantry of 565 mm / respix
sizez_im     = 45   # depth of 176.4 mm / respix thus 45 pix, but we use a cube bounding box
size_border  = 55

#src   = 'toto.bin'
#trg   = 'mire'
#kind  = 'int' #'int'  # or 'float'
Ntot  = os.path.getsize(src) // 16
#Nstep = 2
#rnd   = 10

nclean = 0
if kind == 'int':
    xg1 = array([], 'uint16')
    yg1 = array([], 'uint16')
    zg1 = array([], 'uint16')
    xg2 = array([], 'uint16')
    yg2 = array([], 'uint16')
    zg2 = array([], 'uint16')
elif kind == 'float':
    xg1 = array([], 'float32')
    yg1 = array([], 'float32')
    zg1 = array([], 'float32')
    xg2 = array([], 'float32')
    yg2 = array([], 'float32')
    zg2 = array([], 'float32')

# main loop
for istep in xrange(Nstep):
    n_start = int(float(Ntot) / Nstep * istep + 0.5)
    n_stop  = int(float(Ntot) / Nstep * (istep+1) + 0.5)
    n       = n_stop - n_start
    print '\n## iter %i / %i   ' % (istep, Nstep), '%i - %i' % (n_start, n_stop)
    
    # read data
    t = time()
    idc1 = zeros((n), 'int32')
    idd1 = zeros((n), 'int32')
    idc2 = zeros((n), 'int32')
    idd2 = zeros((n), 'int32')
    kernel_listmode_open_subset_ID_int(idc1, idd1, idc2, idd2, n_start, n_stop, src)
    print 'Read data', time_format(time()-t)

    # precompute all global position
    t = time()
    x1   = zeros((n), 'float32')
    y1   = zeros((n), 'float32')
    z1   = zeros((n), 'float32')
    x2   = zeros((n), 'float32')
    y2   = zeros((n), 'float32')
    z2   = zeros((n), 'float32')
    kernel_allegro_idtopos(idc1, idd1, x1, y1, z1, idc2, idd2, x2, y2, z2, respix, sizexy_space, sizez_space, rnd)
    del idc1, idd1, idc2, idd2
    print 'Convert ID to pos', time_format(time()-t)

    # precompute all entry-exit SRM point
    t = time()
    enable = zeros((n), 'int32')
    kernel_pet3D_SRM_raycasting(x1, y1, z1, x2, y2, z2, enable, size_border, sizexy_im, sizez_im)
    print 'Compute SRM entry-exit points', time_format(time()-t)

    # clean all LORs
    t = time()
    N = enable.sum()
    nclean += N
    if kind == 'int':
        xi1 = zeros((N), 'int32')
        yi1 = zeros((N), 'int32')
        zi1 = zeros((N), 'int32')
        xi2 = zeros((N), 'int32')
        yi2 = zeros((N), 'int32')
        zi2 = zeros((N), 'int32')
        kernel_pet3D_SRM_clean_LOR_int(enable, x1, y1, z1, x2, y2, z2, xi1, yi1, zi1, xi2, yi2, zi2)
        xi1 = xi1.astype('uint16')
        yi1 = yi1.astype('uint16')
        zi1 = zi1.astype('uint16')
        xi2 = xi2.astype('uint16')
        yi2 = yi2.astype('uint16')
        zi2 = zi2.astype('uint16')
    elif kind == 'float':
        xi1 = zeros((N), 'float32')
        yi1 = zeros((N), 'float32')
        zi1 = zeros((N), 'float32')
        xi2 = zeros((N), 'float32')
        yi2 = zeros((N), 'float32')
        zi2 = zeros((N), 'float32')
        kernel_pet3D_SRM_clean_LOR_float(enable, x1, y1, z1, x2, y2, z2, xi1, yi1, zi1, xi2, yi2, zi2)
    
    del x1, y1, z1, x2, y2, z2
    print 'Clean outliers LORs', time_format(time()-t)
    
    # append data
    xg1 = concatenate((xg1, xi1))
    yg1 = concatenate((yg1, yi1))
    zg1 = concatenate((zg1, zi1))
    xg2 = concatenate((xg2, xi2))
    yg2 = concatenate((yg2, yi2))
    zg2 = concatenate((zg2, zi2))
    del xi1, yi1, zi1, xi2, yi2, zi2
    
t = time()
xg1.tofile('%s.x1' % trg)
yg1.tofile('%s.y1' % trg)
zg1.tofile('%s.z1' % trg)
xg2.tofile('%s.x2' % trg)
yg2.tofile('%s.y2' % trg)
zg2.tofile('%s.z2' % trg)

print 'Save data', time_format(time()-t), nclean, 'lines'


