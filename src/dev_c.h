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

// Raytracer to Emanuelle BRARD - AMELL
int dev_AMELL(int* voxel_ind, int nvox, float* voxel_val, int nvox2, int dimx, int dimy, int dimz,
			  float x1, float y1, float z1,
			  float x2, float y2, float z2);

void dev_MSPS_build(float* org_act, int nact, int* ind, int nind);

void dev_MSPS_gen(float* msv, int nmsv, int* msi, int nmsi, int* nk, int nnk, int* indk, int nindk,
				  float* X, int sx, float* Y, int sy, float* Z, int sz, int* step, int nstep,
				  int npoint, int seed, int nz, int ny, int nx);

void dev_MSPS_acc(float* im, int nz, int ny, int nx,
				  float* x, int sx, float* y, int sy, float* z, int sz);

void dev_MSPS_naive(float* act, int nact, int* indact, int inact,
					float* X, int sx, float* Y, int sy, float* Z, int sz,
					int* step, int nstep,
					int npoint, int seed, int nz, int ny, int nx);
