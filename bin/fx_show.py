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
usage    = progname + ' filename'
topic    = 'Display image and volume in FIREwork format (.png, .im and .vol)'
p        = optparse.OptionParser(usage, description=topic)
p.add_option('--slices', action='store_true', default=False, help='Slices viewer mode')
p.add_option('--mip',    action='store_true', default=False, help='Mip viewer mode')
#p.add_option('--nb_crystals', type='int',     default=289,  help='Number of crystals')

(options, args) = p.parse_args()
if len(args) < 1:
    print topic
    print ''
    print 'usage:', usage
    print ''
    print 'please run "' + progname + ' -h" for detailed options'
    sys.exit()

from firework import *
from os.path  import splitext

filename  = args[0]
name, ext = splitext(filename)

if options.mip:
    if ext == '.im':
        im = image_open(filename)
        image_show(im)
    elif ext == '.vol':
        vol = volume_open(filename)
        volume_show_mip(vol)
    elif ext == '.png':
        im = image_open(filename)
        image_show(im)
    else:
        print 'File format unknow by FIREwork!'
elif options.slices:
    if ext == '.vol':
        vol   = volume_open(filename)
        minv  = vol.min()
        scale = float(vol.max()) - minv
        vol   = (vol - minv) / scale
        vol   = volume_flip_ud(vol)
        volume_show_slices(vol)
    else:
        print 'File format unknow by FIREwork!'

else:
    if ext == '.vol':
        vol   = volume_open(filename)
        minv  = vol.min()
        scale = float(vol.max()) - minv
        vol   = (vol - minv) / scale
        vol   = volume_flip_ud(vol)
        volume_show_ct(vol)
    else:
        print 'File format unknow by FIREwork!'
    
