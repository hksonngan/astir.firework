// This file is part of FIREwire
// 
// FIREwire is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FIREwire is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FIREwire.  If not, see <http://www.gnu.org/licenses/>.
//
// FIREwire Copyright (C) 2008 - 2010 Julien Bert 

%module kernel

%{
#define SWIG_FILE_WITH_INIT
#include "kernel.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

void omp_vec_square(float* INPLACE_ARRAY1, int DIM1);

void kernel_draw_voxels(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float gamma, float thres);
void kernel_draw_voxels_edge(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float thres);

void kernel_draw_2D_line_DDA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_line_BLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_line_WLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_line_WALA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_line_DDAA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);

void kernel_draw_2D_alllines_BLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1);
void kernel_pet2D_square_gen_sim_ID(int* INPLACE_ARRAY1, int DIM1, float posx, float posy, float alpha, int nx);
void kernel_build_2D_SRM_BLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int wx);
void kernel_pet2D_EMML_iter(float* INPLACE_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY1, int DIM1, float* INPLACE_ARRAY1, int int DIM1, int* IN_ARRAY1, int DIM1);

void kernel_pet2D_ring_build_SM(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int x2, int y1, int y2, int nx);

void kernel_draw_3D_line_DDA(float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3, int x1, int y1, int z1, int x2, int y2, int z2, float val);