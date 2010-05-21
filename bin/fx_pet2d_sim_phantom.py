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
usage    = progname + ' LOR.txt <activities.tif> --nb_crystals=289 --nb_particles=100000 --rand_seed=10 --mode=bin'
topic    = 'Create LORs simulated from a phantom indside a 2D PET ring scan'
p        = optparse.OptionParser(usage, description=topic)
p.add_option('--nb_crystals',  type='int',     default=289,    help='Number of crystals')
p.add_option('--nb_particles', type='int',     default=100000, help='Number of particles')
p.add_option('--rand_seed',    type='int',     default=10,     help='Value of random seed')
p.add_option('--mode',         type='string',  default='bin',  help='Date mode, bin or list-mode')

(options, args) = p.parse_args()
if len(args) < 1 or len(args) > 2:
    print topic
    print ''
    print 'usage:', usage
    print ''
    print 'please run "' + progname + ' -h" for detailed options'
    sys.exit()
else:
    lor_name = args[0]
    if len(args) == 2: name_act = args[1]
    else:              name_act = None
    
from firework import *

LOR_val, LOR_id1, LOR_id2, image = pet2D_ring_simu_circle_phantom(options.nb_crystals, options.nb_particles, options.rand_seed, options.mode)
nlor = len(LOR_id1)
print 'Number of LOR:', nlor

if options.mode == 'bin':
    import pickle
    f = open(lor_name, 'w')
    pickle.dump([LOR_val, LOR_id1, LOR_id2], f)
    f.close()
else:
    f = open(lor_name, 'w')
    for n in xrange(nlor):
        f.write('%i %i\n' % (LOR_id1[n], LOR_id2[n]))
    f.close()
    
if name_act is not None:
    image_write(image, name_act)
