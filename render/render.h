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
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <GL/gl.h>
#include <assert.h>

// Volume rendering
void render_gl_draw_pixels(float* mapr, int him, int wim, float* mapg,
						   int himg, int wimg, float* mapb, int himb, int wimb);

void render_image_color(float* im, int him, int wim, float* mapr, int him1,
						int wim1, float* mapg, int him2, int wim2, float* mapb,
						int him3, int wim3,	float* lutr, int him4, float* lutg,
						int him5, float* lutb, int him6);

void render_volume_mip(float* vol, int nz, int ny, int nx, float* mip,
					   int wim, int him, float alpha, float beta, float scale);

void render_volume_surf(float* vol, int nz, int ny, int nx, float* mip,
						int him, int wim, float alpha, float beta, float scale, float th);

// 2D drawing line
void render_line_2D_DDA(float* mat, int wy, int wx, int x1, int y1,
						int x2, int y2, float val);

void render_lines_2D_DDA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1,
						 int* X2, int nx2, int* Y2, int ny2);

void render_lines_2D_DDA_fixed(float* mat, int wy, int wx, int* X1, int nx1,
							   int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);

void render_line_2D_BLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2,
						float val);

void render_lines_2D_BLA(float* mat, int wy, int wx, int* X1, int nx1,
						 int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);

void render_line_2D_WLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2,
						float val);

void render_lines_2D_WLA(float* mat, int wy, int wx, int* X1, int nx1,
						 int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);

void render_line_2D_WALA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2,
						 float val);

void render_lines_2D_WALA(float* mat, int wy, int wx, int* X1, int nx1,
						  int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2);

void render_lines_2D_SIDDON(float* mat, int wy, int wx, float* X1, int nx1,
							float* Y1, int ny1, float* X2, int nx2,
							float* Y2, int ny2, int res, int b, int matsize);

// 3D drawing line
void render_line_3D_DDA(float* mat, int wz, int wy, int wx,
						int x1, int y1, int z1, int x2, int y2, int z2, float val);

void render_line_3D_BLA(float* mat, int wz, int wy, int wx,
						int x1, int y1, int z1, int x2, int y2, int z2, float val);

