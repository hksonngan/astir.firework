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

void filter_c_2d_median(float* im, int ny, int nx, float* res, int nyr, int nxr, int w);
void filter_c_3d_median(float* im, int nz, int ny, int nx, float* res, int nzr, int nyr, int nxr, int w);
void filter_c_2d_adaptive_median(float* im, int ny, int nx, float* res, int nyr, int nxr, int w, int wmax);
void filter_c_3d_adaptive_median(float* im, int nz, int ny, int nx,
							     float* res, int nzr, int nyr, int nxr, int w, int wmax);

