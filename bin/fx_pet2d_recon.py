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
usage    = progname + ' LOR.txt SM.txt image.tif --nb_crystals=289 --maxit=8 --show --MPI --CUDA'
topic    = ' dfdsf sdfs'
p        = optparse.OptionParser(usage, description=topic)
p.add_option('--nb_crystals', type='int',          default=289,   help='Number of crystals')
p.add_option('--maxit',       type='int',          default=8,     help='Number of iterations')
p.add_option('--show',        action='store_true', default=False, help='Show the result')
p.add_option('--MPI',         action='store_true', default=False, help='Run with several CPUs')
p.add_option('--CUDA',        action='store_true', default=False, help='Run on GPU')

(options, args) = p.parse_args()
if len(args) < 3:
    print topic
    print ''
    print 'usage:', usage
    print ''
    print 'please run "' + progname + ' -h" for detailed options'
    sys.exit()

from firework import *
from math     import sqrt
from time     import time
import pickle

lor_name = args[0]
sm_name  = args[1]
im_name  = args[2]

# open SM
f  = open(sm_name, 'r')
SM = pickle.load(f)
f.close()
npix = SM.size
nx   = sqrt(npix)

# open LOR
f = open(lor_name, 'r')
LOR_val, LOR_id1, LOR_id2 = pickle.load(f)
f.close()
nlor = LOR_val.size

pref = ['', 'k', 'M', 'G', 'T'] 
print '## PET 2D - EMML ##'
print 'number of LORs: %i' % nlor
print 'image size:     %ix%i (%i pixels)' % (nx, nx, npix)

if options.MPI and not options.CUDA:
    from mpi4py import MPI
    from math   import log

    ncpu = MPI.COMM_WORLD.Get_size()
    myid = MPI.COMM_WORLD.Get_rank()
    main_node = 0
    if myid == main_node:
        print 'MPI version'
        print 'number of CPUs: %i' % ncpu
        mem    = nlor*npix*4*ncpu + 3*nlor*4*ncpu + 4*npix*4*ncpu
        iemem  = int(log(mem) // log(1e3))
        mem   /= (1e3 ** iemem)
        print 'mem estimation: %5.2f %sB' % (mem, pref[iemem])
    
    N_start = int(round(float(npix) / ncpu * myid))
    N_stop  = int(round(float(npix) / ncpu * (myid+1)))
    #nloclor = N_stop - N_start + 1

    # build SRM
    SRM  = zeros((nlor, npix),    'float32')
    kernel_pet2D_ring_LOR_SRM_BLA(SRM, LOR_val, LOR_id1, LOR_id2, options.nb_crystals)
    
    ### iteration loop
    im   = image_1D_projection(SRM, 'y')
    res  = zeros((npix), 'float32')
    mask = zeros((npix), 'float32')
    mask[N_start:N_stop] = 1.0
    
    if myid == main_node: t1 = time()
    for ite in xrange(options.maxit):
        kernel_pet2D_EMML_iter_MPI(SRM, SM, im, LOR_val, N_start, N_stop)
        # gather image
        im     *= mask
        res[:]  = 0.0

        MPI.COMM_WORLD.Allreduce([im, MPI.FLOAT], [res, MPI.FLOAT], op=MPI.SUM)
        im      = res.copy()
        if myid == main_node: print 'ite', ite

        MPI.COMM_WORLD.Barrier()
        
    if myid == main_node:
        t2 = time()
        im = im.reshape((nx, nx))
        if options.show: image_show(im)
        image_write(im, im_name)

        print 'Running time', t2-t1, 's'
else:
    if options.CUDA:
        from math import log
        mem    = nlor*npix*4 + 2*npix*4 + 3*nlor*4 
        iemem  = int(log(mem) // log(1e3))
        mem   /= (1e3 ** iemem)
        print 'GPU version'
        print 'mem estimation: %5.2f %sB' % (mem, pref[iemem])

        # build SRM
        SRM  = zeros((nlor, npix),    'float32')
        kernel_pet2D_ring_LOR_SRM_BLA(SRM, LOR_val, LOR_id1, LOR_id2, options.nb_crystals)
        # compute image
        im  = image_1D_projection(SRM, 'y')
        # reconstruction
        t1 = time()
        kernel_pet2D_EMML_cuda(SRM, im, LOR_val, SM, options.maxit)
        t2 = time()

        im = im.reshape((nx, nx))
        if options.show: image_show(im)
        image_write(im, im_name)

        print 'Running time', t2-t1, 's'

    else:
        # build SRM
        SRM  = zeros((nlor, npix),    'float32')
        kernel_pet2D_ring_LOR_SRM_BLA(SRM, LOR_val, LOR_id1, LOR_id2, options.nb_crystals)

        ### iteration loop
        im = image_1D_projection(SRM, 'y')

        t1 = time()
        for ite in xrange(options.maxit):
            kernel_pet2D_EMML_iter(SRM, SM, im, LOR_val)
            print 'ite', ite
        t2 = time()

        im = im.reshape((nx, nx))
        if options.show: image_show(im)
        image_write(im, im_name)

        print 'Running time', t2-t1, 's'


