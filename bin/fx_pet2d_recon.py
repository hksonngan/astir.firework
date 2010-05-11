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
usage    = progname + ' LOR.txt SM.txt image.tif --nb_crystals=289 --maxit=8'
topic    = ' dfdsf sdfs'
p        = optparse.OptionParser(usage, description=topic)
p.add_option('--nb_crystals', type='int',     default=289,  help='Number of crystals')
p.add_option('--maxit',       type='int',     default=8,    help='Number of iterations')

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

# build SRM
SRM = zeros((nlor, npix), 'float32')
kernel_pet2D_ring_LOR_SRM_BLA(SRM, LOR_val, LOR_id1, LOR_id2, options.nb_crystals)

### iteration loop
im = image_1D_projection(SRM, 'y')
t1 = time()
for ite in xrange(options.maxit):
    kernel_pet2D_EMML_iter(SRM, SM, im, LOR_val)
    print 'ite', ite
t2 = time()

im = im.reshape((nx, nx))
image_write(im, im_name)

print 'Running time', t2-t1, 's'


