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

#include <stdlib.h>
#include <stdio.h>

// Utils function
void inkernel_quicksort(float* vec, int m, int n);
void inkernel_bubblesort(float* vec, int n);
int inkernel_mono(int i, int j);
float inkernel_randf();
float inkernel_randgf(float mean, float std);
void inkernel_randg2f(float mean, float std, float* z0, float* z1);

// Volume rendering
void kernel_draw_voxels(int* posxyz, int npos, float* val, int nval, float* valthr, int nthr, float gamma, float thres);
void kernel_draw_voxels_edge(int* posxyz, int npos, float* val, int nval, float* valthr, int nthr, float thres);
void kernel_draw_pixels(float* mapr, int him, int wim, float* mapg, int himg, int wimg, float* mapb, int himb, int wimb);
void kernel_color_image(float* im, int him, int wim,
						float* mapr, int him1, int wim1, float* mapg, int him2, int wim2, float* mapb, int him3, int wim3,
						float* lutr, int him4, float* lutg, int him5, float* lutb, int him6);
void kernel_mip_volume_rendering(float* vol, int nz, int ny, int nx, float* mip, int wim, int him, float alpha, float beta, float scale);
void kernel_volume_rendering(float* vol, int nz, int ny, int nx, float* mip, int him, int wim, float alpha, float beta, float scale, float th);

// 2D drawing line
void kernel_draw_2D_line_DDA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_DDA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_lines_DDA_fixed(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_line_BLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_BLA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_line_WLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_WLA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_line_WALA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val);
void kernel_draw_2D_lines_WALA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);
void kernel_draw_2D_lines_SIDDON(float* mat, int wy, int wx, float* X1, int nx1, float* Y1, int ny1, float* X2, int nx2, float* Y2, int ny2, int res, int b, int matsize);

// 3D drawing line
void kernel_draw_3D_line_DDA(float* mat, int wz, int wy, int wx, int x1, int y1, int z1, int x2, int y2, int z2, float val);
void kernel_draw_3D_line_BLA(float* mat, int wz, int wy, int wx, int x1, int y1, int z1, int x2, int y2, int z2, float val);

// Vector/Matrix operations
int kernel_vector_nonzeros(float* mat, int ni);
int kernel_matrix_nonzeros(float* mat, int ni, int nj);
void kernel_matrix_nonzeros_rows(float* mat, int ni, int nj, int* rows, int nrows);
void kernel_matrix_sumcol(float* mat, int ni, int nj, float* im, int npix);

// Filteration
void kernel_matrix_lp_H(float* mat, int nk, int nj, int ni, float fc, int order);
void kernel_flatvolume_gaussian_filter_3x3x3(float* mat, int nmat, int nk, int nj, int ni);

