// This file is part of FIREwork
// 
// FIREwork is free software: you can redistribute it and/or modify
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
// along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
//
// FIREwork Copyright (C) 2008 - 2011 Julien Bert 

void mc_cuda(float* vol, int nz, int ny, int nx,
			 float* E, int nE, float* dx, int ndx, float* dy, int ndy, float* dz, int ndz,
			 float* px, int npx, float* py, int npy, float* pz, int npz,
			 int nparticles, int seed);
void mc_get_cs_cuda(float* CS, int ncs, float* E, int nE, int mat);

int mc_disk_detector(float* x, int nx, float* y, int ny, float* E, int nE, float* resE, int nrE,
					 int rad, int posx, int posy);
