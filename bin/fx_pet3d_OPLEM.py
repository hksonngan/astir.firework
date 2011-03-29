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
# FIREwork Copyright (C) 2008 - 2011 Julien Bert 

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
p.add_option('--rayproj', action='store', type='choice',  choices=['siddon', 'ddaell'], default='ddaell',
             help='"siddon" Siddon ray-projector, "ddaell" DDA-ELL ray-projector')
p.add_option('--cuda',    action='store_true', default=False, help='Run with GPU support')

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
Nsub      = options.Nsub
cuton     = options.cuton
cutoff    = options.cutoff
NMname    = options.NM
AMname    = options.AM
nxy       = options.nxy
nz        = options.nz
rayproj   = options.rayproj
cuda      = options.cuda

from firework import *
from numpy    import *
from time     import time

print '=========================================='
print '==       3D OPLEM Reconstruction        =='
print '=========================================='

print 'parameters:'
print 'filename', src
print 'output', output
print 'Nsub', Nsub
print 'cuton', cuton
print 'cutoff', cutoff
print 'Volume %ix%ix%i' % (nxy, nxy, nz)
print 'Rayproj', rayproj
print 'Cuda', cuda
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

# read attenuation matrix
if AMname != 'None':
    AM = volume_open(AMname)

# create directory
os.mkdir(output)

# read data
t = time()
if rayproj == 'ddaell':
    xi1  = zeros((ntot), 'uint16')
    yi1  = zeros((ntot), 'uint16')
    zi1  = zeros((ntot), 'uint16')
    xi2  = zeros((ntot), 'uint16')
    yi2  = zeros((ntot), 'uint16')
    zi2  = zeros((ntot), 'uint16')
    kernel_listmode_open_subset_xyz_int(xi1, yi1, zi1, xi2, yi2, zi2, cuton, cutoff, src)
else:
    xf1 = zeros((ntot), 'float32')
    yf1 = zeros((ntot), 'float32')
    zf1 = zeros((ntot), 'float32')
    xf2 = zeros((ntot), 'float32')
    yf2 = zeros((ntot), 'float32')
    zf2 = zeros((ntot), 'float32')
    kernel_listmode_open_subset_xyz_float(xf1, yf1, zf1, xf2, yf2, zf2, cuton, cutoff, src)
print 'Read data'
print '...', time_format(time()-t)

# init im
im  = ones((nz, nxy, nxy), 'float32')

tg  = time()
# OPLEM cuda version
if cuda:
    GPU = 1
    if AMname == 'None':
        kernel_pet3D_OPLEM_cuda(xi1, yi1, zi1, xi2, yi2, zi2, im, NM, Nsub, GPU)
    else:
        kernel_pet3D_OPLEM_att_cuda(xi1, yi1, zi1, xi2, yi2, zi2, im, NM, AM, Nsub, GPU)
else:
# OPLEM
    if rayproj == 'ddaell':
        if AMname == 'None':
            kernel_pet3D_OPLEM(xi1, yi1, zi1, xi2, yi2, zi2, im, NM, Nsub)
        else:
            print 'ddaell att'
            kernel_pet3D_OPLEM_att(xi1, yi1, zi1, xi2, yi2, zi2, im, NM, AM, Nsub)
    elif rayproj == 'siddon':
        if AMname == 'None':
            print 'No code for that!!'
        else:
            #border = 55 # Allegro
            border = 50 # Discovery
            kernel_pet3D_OPLEM_sid_att(xf1, yf1, zf1, xf2, yf2, zf2, im, NM, AM, Nsub, border)

print 'Running time is', time_format(time()-tg)

# save image
volume_write(im, output + '/res_volume.vol')
mip = volume_mip(im)
image_write(mip, output + '/mip_t.png')
mip = volume_mip(im, 'y')
image_write(mip, output + '/mip_c.png')
