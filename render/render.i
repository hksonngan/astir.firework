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

%module render_c

%{
#define SWIG_FILE_WITH_INIT
#include "render.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

// Volume rendering
void render_gl_draw_pixels(float* IN_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY2,
						   int DIM1, int DIM2,	float* IN_ARRAY2, int DIM1, int DIM2);

void render_image_color(float* IN_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY2, int DIM1, int DIM2,
						float* IN_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY2, int DIM1, int DIM2,
						float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1);

void render_volume_mip(float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
					   float* INPLACE_ARRAY2, int DIM1, int DIM2,
					   float alpha, float beta, float scale);

void render_volume_surf(float* IN_ARRAY3, int DIM1, int DIM2, int DIM3,
						float* INPLACE_ARRAY2, int DIM1, int DIM2,
						float alpha, float beta, float scale, float th);

// 2D line-drawing
void render_line_2D_DDA(float* INPLACE_ARRAY2, int DIM1, int DIM2,
						int x1, int y1, int x2, int y2, float val);

void render_lines_2D_DDA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1,
						 int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);

void render_lines_2D_DDA_fixed(float* INPLACE_ARRAY2, int DIM1, int DIM2,
							   int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1,
							   int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);

void render_line_2D_BLA(float* INPLACE_ARRAY2, int DIM1, int DIM2,
						int x1, int y1, int x2, int y2, float val);

void render_lines_2D_BLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1,
						 int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);

void render_line_2D_WLA(float* INPLACE_ARRAY2, int DIM1, int DIM2,
						int x1, int y1, int x2, int y2, float val);

void render_lines_2D_WLA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1,
						 int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);

void render_line_2D_WALA(float* INPLACE_ARRAY2, int DIM1, int DIM2,
						 int x1, int y1, int x2, int y2, float val);

void render_lines_2D_WALA(float* INPLACE_ARRAY2, int DIM1, int DIM2, int* IN_ARRAY1, int DIM1,
						  int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1);

void render_lines_2D_SIDDON(float* INPLACE_ARRAY2, int DIM1, int DIM2, float* IN_ARRAY1, int DIM1,
							float* IN_ARRAY1, int DIM1, float* IN_ARRAY1, int DIM1,
							float* IN_ARRAY1, int DIM1, int res, int b, int matsize);

// 3D line-drawing
void render_line_3D_DDA(float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
						int x1, int y1, int z1, int x2, int y2, int z2, float val);

void render_line_3D_BLA(float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3,
						int x1, int y1, int z1, int x2, int y2, int z2, float val);




