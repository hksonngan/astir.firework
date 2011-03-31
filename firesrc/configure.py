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

from os.path import isfile, basename, join
from os      import environ, system
import optparse, sys

progname = basename(sys.argv[0])
usage    = progname + ' [options]'
topic    = 'Build configuration'
p        = optparse.OptionParser(usage, description=topic)
p.add_option('--nogpu', action='store_true', default=False, help='Build without GPU support')
(options, args) = p.parse_args()
if not options.nogpu: cuda = True
else:                 cuda = False
opengl = True

def findfilepython(name):
    for dirname in sys.path:
        fullpath = join(dirname, name)
        if isfile(fullpath): return True
    return False

def findfileos(name):
    for dirname in environ['PATH'].split(':'):
        fullpath = join(dirname, name)
        if isfile(fullpath): return True
    return False

def error(name):
    print '[ERROR]', name, 'was not found check your system'
    sys.exit()

def txtgreen(txt):  return '\033[0;32m%s\033[m' % txt
def txtred(txt):    return '\033[0;31m%s\033[m' % txt
def txtyellow(txt): return '\033[0;33m%s\033[m' % txt
        
print txtyellow('.:: Build options ::.')

if cuda: print 'GPU support:', txtgreen('yes')
else:    print 'GPU support:', txtred('no')
if opengl: print 'Viewer Opengl:', txtgreen('yes')
else:      print 'Viewer Opengl:', txtred('no')

print txtyellow('.:: Check build install ::.')

# OpenGL
if isfile('/usr/include/GL/gl.h'): print 'checking for OpenGL...', txtgreen('yes')
else:                              print 'checking for OpenGL...', txtred('no')
# gcc
if findfileos('gcc'):  print 'checking for gcc...', txtgreen('yes')
else:
    print 'checking for gcc...', txtred('no')
    error('gcc')
# nvcc
if findfileos('nvcc'): print 'checking for nvcc...', txtgreen('yes')
else:
    print 'checking for nvcc...', txtred('no')
    if cuda:
        print 'if you do not use GPU support use option --nogpu'
        error('nvcc')
# swig
if findfileos('swig'): print 'checking for swig...', txtgreen('yes')
else:
    print 'checking for swig...', txtred('no')
    error('swig')
# sphinx
if findfileos('sphinx-build'): print 'checking for sphinx...', txtgreen('yes')
else:
    print 'checking for sphinx...', txtred('no')
    error('sphinx')

print txtyellow('.:: Check Python install ::.')

# Python version
print 'python version...', sys.version.split('\n')[0]
try:
    import numpy
    print 'checking for numpy...', txtgreen('yes')
except:
    print 'checking for numpy...', txtred('no')
    error('numpy')

try:
    import matplotlib
    print 'checking for matplotlib...', txtgreen('yes')
except:
    print 'checking for matplotlib...', txtred('no')
    error('matplotlib')

try:
    import IPython
    print 'checking for ipython...', txtgreen('yes')
except:
    print 'checking for ipython...', txtred('no')
    error('ipython')

try:
    import Image
    print 'checking for PIL...', txtgreen('yes')
except:
    print 'checking for PIL...', txtred('no')
    error('PIL')

try:
    import OpenGL
    print 'checking for pyopengl...', txtgreen('yes')
except:
    print 'checking for pyopengl...', txtred('no')
    error('pyopengl')

print txtyellow('.:: Prepare makefile ::.')

try:
    if cuda: system('cp -f Makefile_all Makefile')
    else:    system('cp -f Makefile_nocuda Makefile')
    print 'generating makefile...', txtgreen('ok')
    print ''
    print 'please run "make && make install"'
except:
    print 'generating makefile... ERROR'
    
    
