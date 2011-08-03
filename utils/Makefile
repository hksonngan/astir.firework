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

include ../firework.conf

# :: C coda module compilation ::
all : _image_c.so _filter_c.so _volume_c.so

_image_c.so : image.o image_wrap.o
	g++ -o _image_c.so -shared image.o image_wrap.o

_filter_c.so : filter.o filter_wrap.o
	g++ -o _filter_c.so -shared filter.o filter_wrap.o

_volume_c.so : volume.o volume_wrap.o
	g++ -o _volume_c.so -shared volume.o volume_wrap.o

# Python wrapping
image_wrap.o : image_wrap.c
	g++ -o image_wrap.o -c -fPIC image_wrap.c -I$(PYTHONINC) -I$(PYTHONLIB)
image_wrap.c : image.i
	swig -python image.i

filter_wrap.o : filter_wrap.c
	g++ -o filter_wrap.o -c -fPIC filter_wrap.c -I$(PYTHONINC) -I$(PYTHONLIB)
filter_wrap.c : filter.i
	swig -python filter.i

volume_wrap.o : volume_wrap.c
	g++ -o volume_wrap.o -c -fPIC volume_wrap.c -I$(PYTHONINC) -I$(PYTHONLIB)
volume_wrap.c : volume.i
	swig -python volume.i

# C code
image.o : image.c
	g++ -o image.o -c image.c -fPIC

filter.o : filter.c
	g++ -o filter.o -c filter.c -fPIC

volume.o : volume.c
	g++ -o volume.o -c volume.c -fPIC



# :: CUDA Code module compilation ::
cuda : _filter_cuda.so

_filter_cuda.so : libfiltercuda.so filter_cuda_wrap.o
	g++ -o _filter_cuda.so -shared filter_cuda_wrap.o -lfiltercuda

# Python wrapping
filter_cuda_wrap.o : filter_cuda_wrap.c
	g++ -o filter_cuda_wrap.o -c -fPIC filter_cuda_wrap.c -I$(PYTHONINC) -I$(PYTHONLIB) -I$(CUDAINC)
filter_cuda_wrap.c : filter_cuda.i
	swig -python filter_cuda.i

# CUDA code
libfiltercuda.so : filter_cuda.cu
	nvcc -o filter_cuda.o -c filter_cuda.cu --compiler-options '-fPIC' -use_fast_math -arch compute_20
	nvcc -o libfiltercuda.so -shared filter_cuda.o -lcufft




clean :
	rm *.pyc || true
	rm *.o || true
	rm *.so || true
	rm image_wrap.c || true
	rm filter_wrap.c || true
	rm volume_wrap.c || true
	rm image_c.py || true
	rm filter_c.py || true
	rm volume_c.py || true
	rm filter_cuda_wrap.c || true
	rm filter_cuda.py || true
