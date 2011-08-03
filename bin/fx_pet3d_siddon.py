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
topic    = '3D PET reconstruction with Siddon'
p        = optparse.OptionParser(usage, description=topic)
p.add_option('--Nite',    type='int',    default=1,       help='Number of iterations (default 1)')
p.add_option('--Nsub',    type='int',    default=1,       help='Number of subsets (default 1)')
p.add_option('--cuton',   type='int',    default=0,       help='Starting number in LOR file (default 0)')
p.add_option('--cutoff',  type='int',    default=1000000, help='Stoping number in LOR file (default 1000000)')
p.add_option('--NM',      type='string', default='None',  help='Normalize matrix path and name (.vol) (default None meaning not normalize)')
p.add_option('--AM',      type='string', default='None',  help='Attenuation matrix path and name (.vol) (default None meaning not attenuation correction)')
p.add_option('--nxy',     type='int',    default=141,     help='Volume size on x and y (transaxial)')
p.add_option('--nz',      type='int',    default=45,      help='Volume size on z (axial)')
p.add_option('--model',   action='store', type='choice', choices=['allegro', 'discovery'], default='allegro', help='["allegro" or "discovery"] Geometry scanner available')

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
model     = options.model

from firework import *
from numpy    import *
from time     import time

print '=========================================='
print '==    PET 3D Reconstruction Siddon      =='
print '=========================================='

print 'parameters:'
print 'filename', src
print 'output', output
print 'Nite', Nite
print 'Nsub', Nsub
print 'cuton', cuton
print 'cutoff', cutoff
print 'NMname', NMname
print 'Volume %ix%ix%i' % (nxy, nxy, nz)

if   model=='allegro':   border = 55
elif model=='discovery': border = 50

# Vars
nvox         = nxy*nxy*nz
ntot         = cutoff-cuton

# read normalize matrix
if NMname == 'None':
    SM = ones((nz * nxy * nxy), 'float32')
else:
    SM  = volume_open(NMname)
    SM  = SM.reshape(SM.size)
    SM /= SM.max()
    SM  = 1 / SM

# read attenuation matrix
if AMname != 'None':
    AM = volume_open(AMname)
    AM = AM.reshape(AM.size)
    
# create directory
os.mkdir(output)

# read data
t   = time()
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
tg    = time()
F     = zeros((nz * nxy * nxy), 'float32')
imsub = ones((nz * nxy * nxy), 'float32')
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
        F       = zeros((nvox), 'float32')
        print '... sub %i / %i' % (isub, Nsub)
        print '...... start %i stop %i N %i' % (n_start, n_stop, n)
        
        # Compute F
        F *= 0.0 # init
        if AMname == 'None':
            kernel_pet3D_LMOSEM_sid(xf1[n_start:n_stop], yf1[n_start:n_stop], zf1[n_start:n_stop], xf2[n_start:n_stop], yf2[n_start:n_stop], zf2[n_start:n_stop], imsub, F, nxy, nz, border)
        else:
            kernel_pet3D_LMOSEM_sid_att(xf1[n_start:n_stop], yf1[n_start:n_stop], zf1[n_start:n_stop], xf2[n_start:n_stop], yf2[n_start:n_stop], zf2[n_start:n_stop], imsub, F, AM, nxy, nz, border)
        print '...... compute EM', time_format(time()-tsub)

        # Normalization
        t = time()
        F *= SM
        print '...... Normalize', time_format(time()-t)
        
        # update
        t      = time()
        imsub *= F
        del F
        print '...... Update sub-image', time_format(time()-t)

        print '...... Subtime', time_format(time()-tsub)

    # save
    buf = imsub.reshape((nz, nxy, nxy))
    mip = volume_mip(buf)
    image_write(mip, output + '/%02i_image.png' % ite)
    volume_write(buf, output + '/%02i_volume.vol' % ite)
    print '... Iter time', time_format(time()-tite)

print 'Running time is', time_format(time()-tg)
