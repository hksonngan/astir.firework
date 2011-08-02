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

/********************************************************
 * Headers
 ********************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void pet_cuda_lmosem(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
					 unsigned short int* z1, int nz1,	unsigned short int* x2, int nx2,
					 unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
					 float* im, int nim1, int nim2, int nim3, float* F, int nf1, int nf2, int nf3,
					 int wim, int ID);

void pet_cuda_lmosem_att(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
						 unsigned short int* z1, int nz1,	unsigned short int* x2, int nx2,
						 unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
						 float* im, int nim1, int nim2, int nim3,
						 float* F, int nf1, int nf2, int nf3,
						 float* mumap, int nmu1, int nmu2, int nmu3, int wim, int ID);

void pet_cuda_oplem(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
					unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
					unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
					float* im, int nim1, int nim2, int nim3,
					float* NM, int NM1, int NM2, int NM3, int Nsub, int ID);

void pet_cuda_oplem_att(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
						unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
						unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
						float* im, int nim1, int nim2, int nim3,
						float* NM, int NM1, int NM2, int NM3,
						float* at, int nat1, int nat2, int nat3,
						int Nsub, int ID);
