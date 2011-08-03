/* -*- C -*-  (not really, but good for syntax highlighting) */
// This file is part of FIREwork
// 
// FIREwork is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FIREwork is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
//
// FIREwork Copyright (C) 2008 - 2011 Julien Bert 

%module filter_c

%{
#define SWIG_FILE_WITH_INIT
#include "filter.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

void filter_c_2d_median(float* IN_ARRAY2, int DIM1, int DIM2,
						float* INPLACE_ARRAY2, int DIM1, int DIM2, int w);

void filter_c_3d_median(float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
						float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3, int w);

void filter_c_2d_adaptive_median(float* IN_ARRAY2, int DIM1, int DIM2,
								 float* INPLACE_ARRAY2, int DIM1, int DIM2, int w, int wmax);

void filter_c_3d_adaptive_median(float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
								 float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3, int w, int wmax);
