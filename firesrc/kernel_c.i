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
#include "kernel_c.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

// Volume rendering
void kernel_draw_voxels(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
						float* IN_ARRAY1, int DIM1, float gamma, float thres);

void kernel_draw_voxels_edge(int* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
							 float* IN_ARRAY1, int DIM1, float thres);

void kernel_draw_pixels(float* IN_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY2, int DIM1, int DIM2,
						float* IN_ARRAY2, int DIM1, int DIM2);

void kernel_color_image(float* IN_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY2, int DIM1, int DIM2,
						float* IN_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY2, int DIM1, int DIM2,
						float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1);

void kernel_mip_volume_rendering(float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
								 float* INPLACE_ARRAY2, int DIM1, int DIM2,
								 float alpha, float beta, float scale);

void kernel_volume_rendering(float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
							 float* INPLACE_ARRAY2, int DIM1, int DIM2,
							 float alpha, float beta, float scale, float th);

// 2D line-drawing
void kernel_draw_2D_line_DDA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_DDA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1,
							  int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);

void kernel_draw_2D_lines_DDA_fixed(float* INPLACE_ARRAY2, int DIM1, int DIM2,
									int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1,
									int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);

void kernel_draw_2D_line_BLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_BLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1,
							  int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);

void kernel_draw_2D_line_WLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_WLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1,
							  int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);

void kernel_draw_2D_line_WALA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_WALA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1,
							   int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);

void kernel_draw_2D_lines_SIDDON(float* INPLACE_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY1, int DIM1,
								 float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
								 float* IN_ARRAY1, int DIM1, int res, int b, int matsize);

// 3D line_drawing
void kernel_draw_3D_line_DDA(float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
							 int x1, int y1, int z1, int x2, int y2, int z2, float val);
void kernel_draw_3D_line_BLA(float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
							 int x1, int y1, int z1, int x2, int y2, int z2, float val);

// Vector/matrix operations
int kernel_vector_nonzeros(float* IN_ARRAY1, int DIM1);
int kernel_matrix_nonzeros(float* IN_ARRAY2, int DIM1, int DIM2);
void kernel_matrix_nonzeros_rows(float* IN_ARRAY2, int DIM1, int DIM2, int* INPLACE_ARRAY1, int DIM1);
void kernel_matrix_sumcol(float* IN_ARRAY2, int DIM1, int DIM2, float* INPLACE_ARRAY1, int DIM1);

// Filteration
void kernel_matrix_lp_H(float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3, float fc, int order);
void kernel_flatvolume_gaussian_filter_3x3x3(float* INPLACE_ARRAY1, int DIM1, int nk, int nj, int ni);



