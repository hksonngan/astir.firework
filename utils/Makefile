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

# Module compilation
all : _image_c.so _filter_c.so _volume_c.so

_image_c.so : image.o image_wrap.o
	gcc -o _image_c.so -shared image.o image_wrap.o

_filter_c.so : filter.o filter_wrap.o
	gcc -o _filter_c.so -shared filter.o filter_wrap.o

_volume_c.so : volume.o volume_wrap.o
	gcc -o _volume_c.so -shared volume.o volume_wrap.o

# Python wrapping
image_wrap.o : image_wrap.c
	gcc -o image_wrap.o -c -fPIC image_wrap.c -I$(PYTHONDIR)/include/python2.7 -I$(PYTHONDIR)/lib/python2.7
image_wrap.c : image.ci
	swig -python image.ci

filter_wrap.o : filter_wrap.c
	gcc -o filter_wrap.o -c -fPIC filter_wrap.c -I$(PYTHONDIR)/include/python2.7 -I$(PYTHONDIR)/lib/python2.7
filter_wrap.c : filter.ci
	swig -python filter.ci

volume_wrap.o : volume_wrap.c
	gcc -o volume_wrap.o -c -fPIC volume_wrap.c -I$(PYTHONDIR)/include/python2.7 -I$(PYTHONDIR)/lib/python2.7
volume_wrap.c : volume.ci
	swig -python volume.ci

# C code
image.o : image.c
	gcc -o image.o -c image.c -fPIC

filter.o : filter.c
	gcc -o filter.o -c filter.c -fPIC

volume.o : volume.c
	gcc -o volume.o -c volume.c -fPIC

clean :
	rm *.pyc || true
	rm *.o || true
	rm *.so || true
	rm image_wrap.c || true
	rm filter_wrap.c || true
	rm volume_wrap.c || true
	rm image_c.py || true
	rm filter_c.py || true
	rm volume_c.py
