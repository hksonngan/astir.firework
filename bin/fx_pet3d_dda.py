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

import optparse, os, sys

progname = os.path.basename(sys.argv[0])
usage    = progname + ' filename output_directory'
topic    = '3D PET reconstruction with DDA'
p        = optparse.OptionParser(usage, description=topic)
p.add_option('--Nite',    type='int',    default=1,       help='Number of iterations (default 1)')
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
Nite      = options.Nite
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
print '==      PET 3D Reconstruction DDA       =='
print '=========================================='

print 'parameters:'
print 'filename', src
print 'output', output
print 'Nite', Nite
print 'Nsub', Nsub
print 'cuton', cuton
print 'cutoff', cutoff
print 'NMname', NMname
print 'AMname', AMname
print 'Volume %ix%ix%i' % (nxy, nxy, nz)

# Vars
ndata = int(1.2 * (2*nxy*nxy + nz*nz)**0.5)
ntot  = cutoff-cuton
print 'ELL buffer size', ndata

# read normalize matrix
if NMname == 'None':
    SM = ones((nz, nxy, nxy), 'float32')
else:
    SM  = volume_open(NMname)
    SM /= SM.max()
    SM  = 1 / SM

# read attenuation matrix
if AMname != 'None':
    AM = volume_open(AMname)
    
# create directory
os.mkdir(output)

# read data
t    = time()
xi1  = zeros((ntot), 'uint16')
yi1  = zeros((ntot), 'uint16')
zi1  = zeros((ntot), 'uint16')
xi2  = zeros((ntot), 'uint16')
yi2  = zeros((ntot), 'uint16')
zi2  = zeros((ntot), 'uint16')
kernel_listmode_open_subset_xyz_int(xi1, yi1, zi1, xi2, yi2, zi2, cuton, cutoff, src)
print 'Read data'
print '...', time_format(time()-t)

# init im
tg    = time()
F     = zeros((nz, nxy, nxy), 'float32')
imsub = ones((nz, nxy, nxy), 'float32')
# Iteration loop
for ite in xrange(Nite):
    print 'Iteration %i' % ite
    tite  = time()
    
    # Subset loop
    for isub in xrange(Nsub):
        tsub    = time()
        n_start = int(float(ntot) / Nsub * isub + 0.5)
        n_stop  = int(float(ntot) / Nsub * (isub+1) + 0.5)
        n       = n_stop - n_start

        print '... sub %i / %i' % (isub, Nsub)
        print '...... start %i stop %i N %i' % (n_start, n_stop, n)
        
        # Compute F
        F *= 0.0 # init
        if AMname == 'None':
            #kernel_pet3D_IM_SRM_ELL_DDA_ON_iter(xi1[n_start:n_stop], yi1[n_start:n_stop], zi1[n_start:n_stop], xi2[n_start:n_stop], yi2[n_start:n_stop], zi2[n_start:n_stop], imsub, F, nxy, ndata)
            kernel_pet3D_LMOSEM_dda(xi1[n_start:n_stop], yi1[n_start:n_stop], zi1[n_start:n_stop], xi2[n_start:n_stop], yi2[n_start:n_stop], zi2[n_start:n_stop], imsub, F, nxy, ndata)
        else:
            kernel_pet3D_LMOSEM_dda_att(xi1[n_start:n_stop], yi1[n_start:n_stop], zi1[n_start:n_stop], xi2[n_start:n_stop], yi2[n_start:n_stop], zi2[n_start:n_stop], imsub, F, AM, nxy, ndata)
        print '...... compute EM', time_format(time()-tsub)

        # Normalization
        t = time()
        F *= SM
        print '...... Normalize', time_format(time()-t)
        
        # update
        t      = time()
        imsub *= F
        print '...... Update sub-image', time_format(time()-t)

        print '...... Subtime', time_format(time()-tsub)
    
    # save image
    mip = volume_mip(imsub)
    image_write(mip, output + '/%02i_image.png' % ite)
    volume_write(imsub, output + '/%02i_volume.vol' % ite)
    print '... Iter time', time_format(time()-tite)

print 'Running time is', time_format(time()-tg)
