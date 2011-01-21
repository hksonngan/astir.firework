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

from time     import time
from struct   import pack
import os, sys

if len(sys.argv) != 3:
    print 'Convert .root Gate coincidence file to bin data'
    print 'fx_convert_Gate_root_to_bin.py source.root target.bin'
    sys.exit()

src = sys.argv[1]
trg = sys.argv[2]
os.system("/home/julien/Gate/root/bin/root -b -q \'/home/julien/recherche/Projet_reconstruction/FIREwork/bin/convert_root2bin.c(\"%s\", \"%s\")\'" % (src, trg))

