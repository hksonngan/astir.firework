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
usage    = progname + ' filename output_directory'
topic    = '3D PET reconstruction with DDA and CUDA'
p        = optparse.OptionParser(usage, description=topic)
#p.add_option('--Nite',    type='int',    default=1,       help='Number of iterations (default 1)')
p.add_option('--Nsub',    type='int',    default=1,       help='Number of subsets (default 1)')
p.add_option('--cuton',   type='int',    default=0,       help='Starting number in LOR file (default 0)')
p.add_option('--cutoff',  type='int',    default=1000000, help='Stoping number in LOR file (default 1000000)')
p.add_option('--NM',      type='string', default='None',  help='Normalize matrix path and name (.vol) (default None meaning not normalize)')
p.add_option('--AM',      type='string', default='None',  help='Attenuation matrix path and name (.vol) (default None meaning not attenuation correction)')
p.add_option('--nxy',     type='int',    default=141,     help='Volume size on x and y (transaxial)')
p.add_option('--nz',      type='int',    default=45,      help='Volume size on z (axial)')

(options, args) = p.parse_args()
if len(args) < 2:
    print topic
    print ''
    print 'usage:', usage
    print ''
    print 'please run "' + progname + ' -h" for detailed options'
    sys.exit()
    
src       = args[0]
output    = args[1]
#Nite      = options.Nite
Nsub      = options.Nsub
cuton     = options.cuton
cutoff    = options.cutoff
NMname    = options.NM
AMname    = options.AM
nxy       = options.nxy
nz        = options.nz

from firework import *
from numpy    import *
from time     import time

print '=========================================='
print '==   PET 3D Reconstruction DDA CUDA     =='
print '=========================================='

print 'parameters:'
print 'filename', src
print 'output', output
#print 'Nite', Nite
print 'Nsub', Nsub
print 'cuton', cuton
print 'cutoff', cutoff
print 'Volume %ix%ix%i' % (nxy, nxy, nz)
print 'Correction:'
print '  Normalization:', NMname
print '  Atenuation:   ', AMname

# Vars
ntot  = cutoff-cuton

# read normalize matrix
if NMname == 'None':
    NM = ones((nz, nxy, nxy), 'float32')
else:
    NM  = volume_open(NMname)
    #NM /= NM.max()

# read attenuation matrix
if AMname != 'None':
    AM = volume_open(AMname)

# create directory
os.mkdir(output)

# read data
t = time()
xi1  = zeros((ntot), 'uint16')
yi1  = zeros((ntot), 'uint16')
zi1  = zeros((ntot), 'uint16')
xi2  = zeros((ntot), 'uint16')
yi2  = zeros((ntot), 'uint16')
zi2  = zeros((ntot), 'uint16')
kernel_listmode_open_subset_xyz_int(xi1, yi1, zi1, xi2, yi2, zi2, cuton, cutoff, src)
print 'Read data'
print '...', time_format(time()-t)

# OPLEM
GPU = 1
im  = ones((nz, nxy, nxy), 'float32')
tg  = time()
if AMname == 'None':
    kernel_pet3D_OPLEM_cuda(xi1, yi1, zi1, xi2, yi2, zi2, im, NM, Nsub, GPU)
else:
    kernel_pet3D_OPLEM_att_cuda(xi1, yi1, zi1, xi2, yi2, zi2, im, NM, AM, Nsub, GPU)
print 'Running time is', time_format(time()-tg)

# save image
mask = volume_mask_cylinder(47, 127, 127, 47, 60)
im *= mask
volume_write(im, output + '/res_volume.vol')
mip = volume_mip(im)
#mip *= image_mask_circle(127, 127, 55)
image_write(mip, output + '/mip_t.png')

mip = volume_mip(im, 'y')
image_write(mip, output + '/mip_c.png')

image_show(mip)
