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
usage    = progname + ' LM.txt SM.txt image.tif --nb_crystals=289 --maxit=8 --subset=10 --show --MPI --CUDA'
topic    = ' PET 2D reconstruction (list-mode OSEM)'
p        = optparse.OptionParser(usage, description=topic)
p.add_option('--nb_crystals', type='int',          default=289,   help='Number of crystals')
p.add_option('--maxit',       type='int',          default=8,     help='Number of iterations')
p.add_option('--subset',      type='int',          default=10,    help='Number of subsets')
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

from math     import sqrt
import pickle

lor_name = args[0]
sm_name  = args[1]
im_name  = args[2]

print '## PET 2D - EMML ##'

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
    #!# kernel_pet2D_ring_LOR_SRM_BLA(SRM, LOR_val, LOR_id1, LOR_id2, options.nb_crystals)
    
    ### iteration loop
    im   = image_1D_projection(SRM, 'y')
    res  = zeros((npix), 'float32')
    mask = zeros((npix), 'float32')
    mask[N_start:N_stop] = 1.0
    
    if myid == main_node: t1 = time()
    for ite in xrange(options.maxit):
        #!# kernel_pet2D_EMML_iter_MPI(SRM, SM, im, LOR_val, N_start, N_stop)
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
        from firework import kernel_pet2D_ring_LM_SRM_BLA, image_1D_projection, kernel_pet2D_LM_EMML_COO_iter
        from firework import image_show, image_write, listmode_nb_events, listmode_open_subset
        from firework import pet2D_ring_simu_build_SRM_COO, prefix_SI, listmode_open_SM
        from kernel   import kernel_matrix_coo_sumcol
        from math     import sqrt
        from numpy    import zeros, array
        from time     import time

        SM     = listmode_open_SM(sm_name)
        npix   = len(SM)
        nx     = int(sqrt(npix))
        nlor   = listmode_nb_events(lor_name)
        mem    = nlor*npix*4 + 2*nlor*4 + 4*npix*4
        print 'number of LORs: %i' % nlor
        print 'image size:     %ix%i (%i pixels)' % (nx, nx, npix)
        print 'mem estimation: %s' % prefix_SI(mem)

        ### iteration loop
        nsub  = options.subset
        gblim = zeros((npix), 'float32')
        tg1   = time()
        for isub in xrange(nsub):
            tsub = time()
            t1   = time()
            N_start  = int(round(float(nlor) / nsub * isub))
            N_stop   = int(round(float(nlor) / nsub * (isub+1)))
            print ' N_start %i N_stop %i' % (N_start, N_stop)
            id1, id2 = listmode_open_subset(lor_name, N_start, N_stop)
            nevents  = len(id1)
            print '## sub %i ##' % isub
            mem      = nevents*npix*4 + 2*nevents*4 + 4*npix*4 
            print ' Open subset', time() - t1, 's with only', nevents, 'events (%s)' % prefix_SI(mem)
            t1 = time()
            SRMval, SRMrow, SRMcol = pet2D_ring_simu_build_SRM_COO(id1, id2, 150000, options.nb_crystals, npix)
            print ' Build SRM', time() - t1, 's'
            #print '   info SRM org  %ix%i=%i' % (nevents, npix, nevents*npix)
            #print '   info SRM COO  %i'       % (3 * len(SRMval))
            #print '   info comp fac %f'       % (nevents*npix / float(3 * len(SRMval)))
            mem    = 3 * len(SRMval) * 4 + 4 * npix * 4
            #print '   info new mem: %s' % prefix_SI(mem)
            t1 = time()
            im = zeros((npix), 'float32')
            kernel_matrix_coo_sumcol(SRMval, SRMcol, im)
            print ' Build im in', time() - t1, 's'
                    
            for ite in xrange(options.maxit):
                t1 = time()
                kernel_pet2D_LM_EMML_COO_iter(SRMval, SRMrow, SRMcol, SM, im, nevents)
                #print '  ite', ite
                print ' Iterations', ite, time() - t1, 's'

            gblim += im
                
            print ' Sub time', time() - tsub, 's'

        print 'Running time', time()-tg1, 's'
            
        gblim = gblim.reshape((nx, nx))
        if options.show: image_show(gblim)
        image_write(gblim, im_name)

        
    else:
        from firework import kernel_pet2D_ring_LM_SRM_BLA, image_1D_projection, kernel_pet2D_LM_EMML_COO_iter
        from firework import image_show, image_write, listmode_nb_events, listmode_open_subset
        from firework import pet2D_ring_simu_build_SRM_COO, prefix_SI, listmode_open_SM
        from kernel   import kernel_matrix_coo_sumcol
        from math     import sqrt
        from numpy    import zeros, array
        from time     import time

        SM     = listmode_open_SM(sm_name)
        npix   = len(SM)
        nx     = int(sqrt(npix))
        nlor   = listmode_nb_events(lor_name)
        mem    = nlor*npix*4 + 2*nlor*4 + 4*npix*4
        print 'number of LORs: %i' % nlor
        print 'image size:     %ix%i (%i pixels)' % (nx, nx, npix)
        print 'mem estimation: %s' % prefix_SI(mem)

        ### iteration loop
        nsub  = options.subset
        gblim = zeros((npix), 'float32')
        tg1   = time()
        for isub in xrange(nsub):
            N_start  = int(round(float(nlor) / nsub * isub))
            N_stop   = int(round(float(nlor) / nsub * (isub+1)))
            id1, id2 = listmode_open_subset(lor_name, N_start, N_stop)
            nevents  = len(id1)
            print '## sub %i ##' % isub
            mem      = nevents*npix*4 + 2*nevents*4 + 4*npix*4 
            SRMval, SRMrow, SRMcol = pet2D_ring_simu_build_SRM_COO(id1, id2, 150000, options.nb_crystals, npix)
            mem      = 3 * len(SRMval) * 4 + 4 * npix * 4
            print '   info mem used: %s' % prefix_SI(mem)
            im = zeros((npix), 'float32')
            kernel_matrix_coo_sumcol(SRMval, SRMcol, im)
                    
            for ite in xrange(options.maxit):
                t1 = time()
                kernel_pet2D_LM_EMML_COO_iter(SRMval, SRMrow, SRMcol, SM, im, nevents)
                print '   Iterations', ite

            gblim += im

        print 'Running time', time()-tg1, 's'
            
        gblim = gblim.reshape((nx, nx))
        if options.show: image_show(gblim)
        image_write(gblim, im_name)




