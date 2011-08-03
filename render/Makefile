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
all : _render_c.so

_render_c.so : render.o render_wrap.o
	g++ -o _render_c.so -shared render.o render_wrap.o -lGL

# Python wrapping
render_wrap.o : render_wrap.c
	g++ -o render_wrap.o -c -fPIC render_wrap.c -I$(PYTHONDIR)/include/python2.7 -I$(PYTHONDIR)/lib/python2.7
render_wrap.c : render.i
	swig -python render.i

# C code
render.o : render.c
	g++ -o render.o -c render.c -fPIC 

clean :
	rm *.pyc || true
	rm *.o || true
	rm *.so || true
	rm render_wrap.c || true
	rm render_c.py || true

