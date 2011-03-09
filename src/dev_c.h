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

// here, put your code!

float rnd_park_miller(int *seed);

void dev_siddon_3D(float* vol, int nz, int ny, int nx, int nlines);

void dev_amanatides_3D(float* vol, int nz, int ny, int nx,
					   float* X0, int nx0, float* Y0, int ny0, float* Z0, int nz0,
					   float* Xe, int nxe, float* Ye, int nye, float* Ze, int nze);

void dev_raypro_3D(float* vol, int nz, int ny, int nx,
				   float* X0, int nx0, float* Y0, int ny0, float* Z0, int nz0,
				   float* DX, int ndx, float* DY, int ndy, float* DZ, int ndz,
				   float* D, int nd);

//void dev_mc_distribution(float* dist, int nz, int ny, int nx, float* res, int nrz, int nry, int nrx, int N);
void dev_mc_distribution(float* dist, int nb, float* small_dist, int small_nb, float* tiny_dist, int tiny_nb,
						 int* ind, int nind, float* res, int nrz, int nrx, int nry, int N);
