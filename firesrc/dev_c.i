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

%module firekernel

%{
#define SWIG_FILE_WITH_INIT
#include "dev_c.h"	
%}

%include "numpy.i"

%init %{
import_array();
%}

// here, put your wrapper
void dev_siddon_3D(float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3, int nlines);

void dev_amanatides_3D(float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
					   float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
					   float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1);

void dev_raypro_3D(float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
				   float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
				   float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
				   float* IN_ARRAY1, int DIM1);

// Raytracer to Emanuelle BRARD - AMELL
int dev_AMELL(int* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int dimx, int dimy, int dimz,
			  float x1, float y1, float z1,
			  float x2, float y2, float z2);


void dev_MSPS_build(float* IN_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1);

void dev_MSPS_gen(float* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1,
				  int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1,
				  float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, int* INPLACE_ARRAY1, int DIM1,
				  int npoint, int seed, int nz, int ny, int nx);

void dev_MSPS_acc(int* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
				  float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1);

void dev_MSPS_naive(float* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1,
					float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int DIM1,
					int* INPLACE_ARRAY1, int DIM1,
					int npoint, int seed, int nz, int ny, int nx);
