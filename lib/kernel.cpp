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
// FIREwork Copyright (C) 2008 - 2010 Julien Bert 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <GL/gl.h>
#include "kernel_cuda.h"

/********************************************************************************
 * PET Scan Allegro      
 ********************************************************************************/

#define pi  3.141592653589
#define twopi 6.283185307179
// Convert ID event from GATE to global position in 3D space
void kernel_allegro_idtopos(int* id_crystal1, int nidc1, int* id_detector1, int nidd1,
							float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
							int* id_crystal2, int nidc2, int* id_detector2, int nidd2,
							float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
							float respix, int sizespacexy, int sizespacez) {
	// NOTE: ref system will be change instead of ref GATE system.
	// We use the image ref system
	// GATE             IMAGE
	//    X             Z
	//    |__ Z         /_ X
	//   /             |
	//  Y              Y
	
	// cst system
	int nic = 22;      // number of crystals along i
	int njc = 29;      // number of crystals along j
	int nd  = 28;      // number of detectors
	float dcz = 6.3;   // delta position of crystal along z (mm)
	float dcx = 4.3;   // delta position of crystal along x (mm)
	float rcz = 88.2;  // org translation of coordinate along z (mm)
	float rcx = 45.15; // org translation of coordinate along x (mm)
	float tsc = 432;   // translation scanner detector along y (mm)
	//float ROI = 565;   // ROI scanner is 565 mm (GANTRY of 56.5 cm)
	//float cxyimage = ROI / respix / 2.0; // center of ROI image along x and y (square image)
	//float czimage  = rcz / respix;       // center of ROI image along z (volume)
	float cxyimage = (float)sizespacexy / 2.0f;
	float czimage = (float)sizespacez / 2.0f;
	float xi, yi, zi, a, newx, newy, newz;
	float cosa, sina;
	int n, ID;
	for (n=0;n<nidc1; ++n) {
		// ID1
		////////////////////////////////
		// global position in GATE space
		ID = id_crystal1[n];
		zi = float(ID / nic) * dcz - rcz;
		xi = float(ID % nic) * dcx - rcx;
		yi = tsc;
		//printf("%f %f\n", zi, xi);
		// rotation accoring ID detector
		a = (float)id_detector1[n] * (-twopi / (float)nd) - pi / 2.0f;
		cosa = cos(a);
		sina = sin(a);
		newx = xi*cosa - yi*sina;
		newy = xi*sina + yi*cosa;
		// change to image org
		newx += cxyimage;           // change origin (left upper corner)
		newy  = (-newy) + cxyimage; // inverse y axis
		newz  = zi + czimage;
		newx /= respix;             // scale factor to match with ROI (image)
		newy /= respix;
		newz /= respix;
		x1[n] = newx;
		y1[n] = newy;
		z1[n] = newz;
		// ID2
		////////////////////////////////
		// global position in GATE space
		ID = id_crystal2[n];
		zi = float(ID / nic) * dcz - rcz;
		xi = float(ID % nic) * dcx - rcx;
		yi = tsc;
		//printf("%f %f\n", zi, xi);
		// rotation accoring ID detector
		a = (float)id_detector2[n] * (-twopi / (float)nd) - pi / 2.0f;
		cosa = cos(a);
		sina = sin(a);
		newx = xi*cosa - yi*sina;
		newy = xi*sina + yi*cosa;
		// change to image org
		newx += cxyimage;           // change origin (left upper corner)
		newy  = (-newy) + cxyimage; // inverse y axis
		newz  = zi + czimage;
		newx /= respix;             // scale factor to match with ROI (image)
		newy /= respix;
		newz /= respix;
		x2[n] = newx;
		y2[n] = newy;
		z2[n] = newz;
	}
}
#undef pi
#undef twopi

// build the list of all LOR in order to compute S matrix
void kernel_allegro_build_all_LOR(unsigned short int* idc1, int n1, unsigned short int* idd1, int n2,
								  unsigned short int* idc2, int n3, unsigned short int* idd2, int n4) {
	int idmax = 22*29;
	int ndete = 28;
	int N = idmax*ndete;
	int Ntot = (N*N - N) / 2;
	int i1, i2;
	unsigned int dc1, dd1, dc2, dd2;
	int ct=0;
	for (i1=0; i1<(N-1); ++i1) {
		dd1 = i1 / idmax;
		dc1 = i1 % idmax;
		for (i2=i1+1; i2<N; ++i2) {
			dd2 = i2 / idmax;
			dc2 = i2 % idmax;
			idc1[ct] = dc1;
			idd1[ct] = dd1;
			idc2[ct] = dc2;
			idd2[ct] = dd2;
			++ct;
		}
	}
}

// SRM Raytracing (transversal algorithm), Compute entry and exit point on SRM of the ray
void kernel_pet2D_SRM_entryexit(float* px, int npx, float* py, int npy, float* qx, int nqx, float* qy, int nqy, int b, int srmsize, int* enable, int nenable) {
	float divx, divy, fsrmsize;
	float axn, ax0, ayn, ay0;
	float amin, amax, buf1, buf2;
	float x1, y1, x2, y2;
	float pxi, pyi, qxi, qyi;
	int i;
		
	b = (float)b;
	fsrmsize = (float)srmsize;

	for (i=0; i<npx; ++i) {
		pxi = px[i];
		pyi = py[i];
		qxi = qx[i];
		qyi = qy[i];
		
		if (pxi == qxi) {divx = 1.0;}
		else {divx = pxi - qxi;}
		if (pyi == qyi) {divy = 1.0;}
		else {divy = pyi - qyi;}
		axn = (fsrmsize + b - qxi) / divx;
		ax0 = (b - qxi) / divx;
		ayn = (fsrmsize + b - qyi) / divy;
		ay0 = (b - qyi) / divy;

		buf1 = ax0;
		if (axn < ax0) {buf1 = axn;}
		buf2 = ay0;
		if (ayn < ay0) {buf2 = ayn;}
		amin = buf2;
		if (buf1 > buf2) {amin = buf1;}
		buf1 = ax0;
		if (axn > ax0) {buf1 = axn;}
		buf2 = ay0;
		if (ayn > ay0) {buf2 = ayn;}
		amax = buf2;
		if (buf1 < buf2) {amax = buf1;}

		x1 = (qxi + amax * (pxi - qxi) - b);
		y1 = (qyi + amax * (pyi - qyi) - b);
		x2 = (qxi + amin * (pxi - qxi) - b);
		y2 = (qyi + amin * (pyi - qyi) - b);

		// format
		if (x1 == fsrmsize) {x1 = fsrmsize-1.0f;}
		if (y1 == fsrmsize) {y1 = fsrmsize-1.0f;}
		if (x2 == fsrmsize) {x2 = fsrmsize-1.0f;}
		if (y2 == fsrmsize) {y2 = fsrmsize-1.0f;}
		// check if ray through the image
		enable[i] = 1;
		if (x1 < 0 || x1 > fsrmsize-1 || y1 < 0 || y1 > fsrmsize-1) {enable[i] = 0;}
		if (x2 < 0 || x2 > fsrmsize-1 || y2 < 0 || y2 > fsrmsize-1) {enable[i] = 0;}
		// check if the ray is > 0
		if (int(x1) == int(x2) && int(y1) == int(y2)) {enable[i] = 0;}
		px[i] = x1;
		py[i] = y1;
		qx[i] = x2;
		qy[i] = y2;
	}
}

// Cleanning LORs outside of ROi based on SRM entry-exit point calculation (return int)
void kernel_pet2D_SRM_clean_entryexit_int(int* enable, int ne, float* x1, int nx1, float* y1, int ny1, float* x2, int nx2, float* y2, int ny2,
									  int* xi1, int nxi1, int* yi1, int nyi1, int* xi2, int nxi2, int* yi2, int nyi2) {
	int i, c;
	c = 0;
	for (i=0; i<nx1; ++i) {
		if (enable[i]) {
			xi1[c] = (int)x1[i];
			yi1[c] = (int)y1[i];
			xi2[c] = (int)x2[i];
			yi2[c] = (int)y2[i];
			++c;
		}
	}
}
// Cleanning LORs outside of ROi based on SRM entry-exit point calculation (return float)
void kernel_pet2D_SRM_clean_entryexit_float(int* enable, int ne, float* x1, int nx1, float* y1, int ny1, float* x2, int nx2, float* y2, int ny2,
									  float* xf1, int nxf1, float* yf1, int nyf1, float* xf2, int nxf2, float* yf2, int nyf2) {
	int i, c;
	c = 0;
	for (i=0; i<nx1; ++i) {
		if (enable[i]) {
			xf1[c] = x1[i];
			yf1[c] = y1[i];
			xf2[c] = x2[i];
			yf2[c] = y2[i];
			++c;
		}
	}
}

// Cleanning LORs outside of ROI based on center LOR position (used by SIDDON to start drawing)
void kernel_pet2D_SRM_clean_LOR_center(float* x1, int nx1, float* y1, int ny1, float* x2, int nx2, float* y2, int ny2,
									   float* xc1, int nxc1, float* yc1, int nyc1, float* xc2, int nxc2, float* yc2, int ncy2, int border, int size_im) {
	int i, c;
	float tx, ty;
	float lx1, ly1, lx2, ly2;
	float lxc1, lyc1, lxc2, lyc2;
	c = 0;
	for (i=0; i<nx1; ++i) {
		lx1 = x1[i];
		ly1 = y1[i];
		lx2 = x2[i];
		ly2 = y2[i];
		tx = (lx2 - lx1) * 0.5 + lx1;
		ty = (ly2 - ly1) * 0.5 + ly1;
		if (tx<border || ty<border) {continue;}
		if (tx>=(border+size_im) || ty>=(border+size_im)) {continue;}
		xc1[c] = lx1;
		yc1[c] = ly1;
		xc2[c] = lx2;
		yc2[c] = ly2;
		++c;
	}
}

// Raytrace SRM matrix with DDA algorithm
void kernel_pet2D_SRM_DDA(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	int length, i, n;
	float flength, val;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy;
	int LOR_ind;
	
	for (i=0; i< nx1; ++i) {
		LOR_ind = i * wx;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		val  = 1 / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
		for (n=0; n<=length; ++n) {
			SRM[LOR_ind + (int)y * width_image + (int)x] = val;
			x = x + xinc;
			y = y + yinc;
		}
	}
}

// Raytrace SRM matrix with DDA algorithm in ELL sparse matrix format
void kernel_pet2D_SRM_ELL_DDA(float* vals, int niv, int njv, int* cols, int nic, int njc, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	int length, i, n;
	float flength, val;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy;
	int LOR_ind;
	val = 1.0f;
	for (i=0; i< nx1; ++i) {
		LOR_ind = i * njv;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		//val  = 1 / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
		for (n=0; n<=length; ++n) {
			vals[LOR_ind + n] = val;
			cols[LOR_ind + n] = (int)y * width_image + (int)x;
			x = x + xinc;
			y = y + yinc;
		}
		cols[LOR_ind + n] = -1; // eof
	}
}

// Raytrace SRM matrix with DDA algorithm with GPU
void kernel_pet2D_SRM_DDA_cuda(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	kernel_pet2D_SRM_DDA_wrap_cuda(SRM, wy, wx, X1, nx1, Y1, ny1, X2, nx2, Y2, ny2, width_image);
}

// OMP version DOES NOT WORK
void kernel_pet2D_SRM_DDA_omp(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	int length, i, n;
	float flength, val;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy;
	int LOR_ind;
	int myid, ncpu;
	int Nstart, Nstop;
#pragma omp parallel num_threads(4)
{
	ncpu = 4; //omp_get_num_threads();
	myid = omp_get_thread_num();
	Nstart = int(float(nx1) / float(ncpu) * float(myid) + 0.5);
	Nstop = int(float(nx1) / float(ncpu) * float(myid + 1) + 0.5);
	printf("myid %i / %i - %i %i\n", myid, ncpu, Nstart, Nstop);
    //#pragma omp parallel for shared(SRM, X1, Y1, X2, Y2) private(i)
	//#pragma omp parallel for private(i)
	for (i=Nstart; i < Nstop; ++i) {
		LOR_ind = i * wx;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		val  = 1 / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
        
		for (n=0; n<=length; ++n) {
			SRM[LOR_ind + (int)y * width_image + (int)x] = val;
			x = x + xinc;
			y = y + yinc;
		}
	}
}
}

// Draw lines in SRM with DDA anti-aliased version 1 pix
void kernel_pet2D_SRM_DDAA(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	int length, i, n;
	float flength;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy, xint, yint;
	int LOR_ind;

	for (i=0; i< nx1; ++i) {
		LOR_ind = i * wx;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
		// line
		for (n=1; n<length; ++n) {
			xint = int(x);
			yint = int(y);
			SRM[LOR_ind + yint * width_image + xint] = (1 - fabs(x - (xint + 0.5)));
			x = x + xinc;
			y = y + yinc;
		}
	}
}

// Draw lines in SRM with DDA anti-aliased version 2 pix 
void kernel_pet2D_SRM_DDAA2(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	int length, i, n;
	float flength;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy, xint, yint, ind;
	float val, vd, vu;
	int LOR_ind;

	for (i=0; i< nx1; ++i) {
		LOR_ind = i * wx;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;

		// first pixel
		xint = int(x);
		yint = int(y);
		val = 1 - fabs(x - (xint + 0.5));
		SRM[LOR_ind + yint * width_image + xint] = val;
		x = x + xinc;
		y = y + yinc;
		// line
		for (n=1; n<length; ++n) {
			xint = int(x);
			yint = int(y);
			ind = LOR_ind + yint * width_image + xint;
			val = 1 - fabs(x - (xint + 0.5));
			vu = (x - xint) * 0.5;
			// vd = 0.5 - vu;
			SRM[ind+1] = vu;
			SRM[ind] = val;
			x = x + xinc;
			y = y + yinc;
		}
		// last pixel
		xint = int(x);
		yint = int(y);
		val = 1 - fabs(x - (xint + 0.5));
		SRM[LOR_ind + yint * width_image + xint] = val;
	}
}

// Draw lines in SRM with DDA anti-aliased version 2 pix, SRM is in ELL sparse matrix format 
void kernel_pet2D_SRM_ELL_DDAA2(float* SRMvals, int niv, int njv, int* SRMcols, int nic, int njc, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	int length, i, n;
	float flength;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy, xint, yint, ind, ind2;
	float val, vd, vu;
	int LOR_ind;

	for (i=0; i< nx1; ++i) {
		LOR_ind = i * njv;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;

		// first pixel
		xint = int(x);
		yint = int(y);
		val = 1 - fabs(x - (xint + 0.5));
		SRMvals[LOR_ind] = val;
		SRMcols[LOR_ind] = yint * width_image + xint;
		//SRM[LOR_ind + yint * width_image + xint] = val;
		x = x + xinc;
		y = y + yinc;
		// line
		for (n=1; n<length; ++n) {
			xint = int(x);
			yint = int(y);
			ind = yint * width_image + xint;
			val = 1 - fabs(x - (xint + 0.5));
			vu = (x - xint) * 0.5;
			// vd = 0.5 - vu;
			ind2 = LOR_ind + 2*n;
			SRMvals[ind2] = vu;
			SRMcols[ind2] = ind + 1;
			SRMvals[ind2 + 1] = val;
			SRMcols[ind2 + 1] = ind;
			//SRM[ind+1] = vu;
			//SRM[ind] = val;
			x = x + xinc;
			y = y + yinc;
		}
		// last pixel
		xint = int(x);
		yint = int(y);
		val = 1 - fabs(x - (xint + 0.5));
		ind2 = LOR_ind + 2*n;
		SRMvals[ind2] = val;
		SRMcols[ind2] = yint * width_image + xint;
		//SRM[LOR_ind + yint * width_image + xint] = val;
	}
}

// Draw lines in SRM by Bresenham's Line Algorithm (modified version 1D)
void kernel_pet2D_SRM_BLA(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image) {
	int x, y, n;
	int x1, y1, x2, y2;
	int dx, dy;
	int xinc, yinc;
	int balance;
	float val;
	int LOR_ind;

	for (n=0; n<nx1; ++n) {
		LOR_ind = n * wx;
		x1 = X1[n];
		y1 = Y1[n];
		x2 = X2[n];
		y2 = Y2[n];

		if (x2 >= x1) {
			dx = x2 - x1;
			xinc = 1;
		} else {
			dx = x1 - x2;
			xinc = -1;
		}
		if (y2 >= y1) {
			dy = y2 - y1;
			yinc = 1;
		} else {
			dy = y1 - y2;
			yinc = -1;
		}
		
		x = x1;
		y = y1;
		if (dx >= dy) {
			val = 1 / (float)dx;
			dy <<= 1;
			balance = dy - dx;
			dx <<= 1;
			while (x != x2) {
				SRM[LOR_ind + y * width_image + x] = val;
				if (balance >= 0) {
					y = y + yinc;
					balance = balance - dx;
				}
				balance = balance + dy;
				x = x + xinc;
			}
			SRM[LOR_ind + y * width_image + x] = val;
		} else {
			val = 1 / (float)dy;
			dx <<= 1;
			balance = dx - dy;
			dy <<= 1;
			while (y != y2) {
				SRM[LOR_ind + y * width_image + x] = val;
				if (balance >= 0) {
					x = x + xinc;
					balance = balance - dy;
				}
				balance = balance + dx;
				y = y + yinc;
			}
			SRM[LOR_ind + y * width_image + x] = val;
		}
	}
}

// Draw lines in SRM by Siddon's Line Algorithm (modified version 1D)
void kernel_pet2D_SRM_SIDDON(float* SRM, int wy, int wx, float* X1, int nx1, float* Y1, int ny1, float* X2, int nx2, float* Y2, int ny2, int matsize) {
	int n, LOR_ind;
	float tx, ty, px, qx, py, qy;
	int ei, ej, u, v, i, j;
	int stepi, stepj;
	float divx, divy, runx, runy, oldv, newv, val, valmax;
	float axstart, aystart, astart, pq, stepx, stepy, startl, initl;

	// random seed
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		LOR_ind = n * wx;
		px = X2[n];
		py = Y2[n];
		qx = X1[n];
		qy = Y1[n];
		initl = (float)rand() / (float)RAND_MAX;
		initl = initl * 0.6 + 0.2; // rnd number between 0.2 to 0.8
		tx = (px-qx) * initl + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * initl + qy;
		ei = int(tx);
		ej = int(ty);
		if (qx-tx>0) {
			u=ei+1;
			stepi=1;
		}
		if (qx-tx<0) {
			u=ei;
			stepi=-1;
		}
		if (qx-tx==0) {
			u=ei;
			stepi=0;
		}
		if (qy-ty>0) {
			v=ej+1;
			stepj=1;
		}
		if (qy-ty<0) {
			v=ej;
			stepj=-1;
		}
		if (qy-ty==0) {
			v=ej;
			stepj=0;
		}
		if (qx==px) {divx=1.0;}
		else {divx = float(qx-px);}
		if (qy==py) {divy=1.0;}
		else {divy = float(qy-py);}
		axstart = (u-px) / divx;
		aystart = (v-py) / divy;
		astart = aystart;
		if (axstart > aystart) {astart = axstart;}
		pq = sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py));
		stepx = fabs(pq / divx);
		stepy = fabs(pq / divy);
		startl = astart * pq;
		valmax = stepx;
		if (stepy < valmax) {valmax = stepy;}
		valmax = valmax + valmax*0.01f;

		// first half-ray
		runx = axstart * pq;
		runy = aystart * pq;
		i = ei;
		j = ej;
		if (runx == startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy == startl) {
			j += stepj;
			runy += stepy;
		}
		oldv = startl;
		while (i>=0 && j>=0 && i<matsize && j<matsize) {
			
			newv = runy;
			if (runx < runy) {newv = runx;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			SRM[LOR_ind + j * matsize + i] = val;
			oldv = newv;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
		}
		// second half-ray
		if (px-tx>0) {stepi=1;}
		if (px-tx<0) {stepi=-1;}
		if (py-ty>0) {stepj=1;}
		if (py-ty<0) {stepj=-1;}
		runx = axstart * pq;
		runy = aystart * pq;
		i = ei;
		j = ej;
		if (runx==startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy==startl) {
			j += stepj;
			runy += stepy;
		}
		SRM[LOR_ind + ej * matsize + ei] = val;
		oldv = startl;
		while (i>=0 && j>=0 && i<matsize && j<matsize) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			SRM[LOR_ind + j * matsize + i] = val;
			oldv = newv;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
		}
	}
}

// Draw lines in SRM by Siddon's Line Algorithm (modified version 1D), SRM is in ELL sparse matrix format
void kernel_pet2D_SRM_ELL_SIDDON(float* SRMvals, int niv, int njv, int* SRMcols, int nic, int njc, float* X1, int nx1, float* Y1, int ny1, float* X2, int nx2, float* Y2, int ny2, int matsize) {
	int n, LOR_ind;
	float tx, ty, px, qx, py, qy;
	int ei, ej, u, v, i, j, ct;
	int stepi, stepj;
	float divx, divy, runx, runy, oldv, newv, val, valmax;
	float axstart, aystart, astart, pq, stepx, stepy, startl, initl;
	// random seed
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		LOR_ind = n * njv;
		ct = 0;
		px = X2[n];
		py = Y2[n];
		qx = X1[n];
		qy = Y1[n];
		initl = (float)rand() / (float)RAND_MAX;
		initl = initl * 0.6 + 0.2; // rnd number between 0.2 to 0.8
		tx = (px-qx) * initl + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * initl + qy;
		ei = int(tx);
		ej = int(ty);
		if (qx-tx>0) {
			u=ei+1;
			stepi=1;
		}
		if (qx-tx<0) {
			u=ei;
			stepi=-1;
		}
		if (qx-tx==0) {
			u=ei;
			stepi=0;
		}
		if (qy-ty>0) {
			v=ej+1;
			stepj=1;
		}
		if (qy-ty<0) {
			v=ej;
			stepj=-1;
		}
		if (qy-ty==0) {
			v=ej;
			stepj=0;
		}
		if (qx==px) {divx=1.0;}
		else {divx = float(qx-px);}
		if (qy==py) {divy=1.0;}
		else {divy = float(qy-py);}
		axstart = (u-px) / divx;
		aystart = (v-py) / divy;
		astart = aystart;
		if (axstart > aystart) {astart = axstart;}
		pq = sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py));
		stepx = fabs(pq / divx);
		stepy = fabs(pq / divy);
		startl = astart * pq;
		valmax = stepx;
		if (stepy < valmax) {valmax = stepy;}
		valmax = valmax + valmax*0.01f;

		// first half-ray
		runx = axstart * pq;
		runy = aystart * pq;
		i = ei;
		j = ej;
		if (runx == startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy == startl) {
			j += stepj;
			runy += stepy;
		}
		oldv = startl;
		while (i>=0 && j>=0 && i<matsize && j<matsize) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			SRMvals[LOR_ind + ct] = val;
			SRMcols[LOR_ind + ct] = j * matsize + i;
			ct++;
			oldv = newv;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
		}

		// second half-ray
		if (px-tx>0) {stepi=1;}
		if (px-tx<0) {stepi=-1;}
		if (py-ty>0) {stepj=1;}
		if (py-ty<0) {stepj=-1;}
		runx = axstart * pq;
		runy = aystart * pq;
		i = ei;
		j = ej;
		if (runx==startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy==startl) {
			j += stepj;
			runy += stepy;
		}
		SRMvals[LOR_ind + ct] = val;
		SRMcols[LOR_ind + ct] = ej * matsize + ei;
		ct++;
		oldv = startl;
		while (i>=0 && j>=0 && i<matsize && j<matsize) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			SRMvals[LOR_ind + ct] = val;
			SRMcols[LOR_ind + ct] = j * matsize + i;
			ct++;
			oldv = newv;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
		}
	}
}

// Draw a list of lines in SRM by Wu's Line Algorithm (modified version 1D)
void kernel_pet2D_SRM_WLA(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int wim) {
	int dx, dy, stepx, stepy, n, LOR_ind;
	int length, extras, incr2, incr1, c, d, i;
	int x1, y1, x2, y2;
	float val;
	for (n=0; n<nx1; ++n) {
		LOR_ind = n * wx;
		x1 = X1[n];
		y1 = Y1[n];
		x2 = X2[n];
		y2 = Y2[n];
	    dy = y2 - y1;
		dx = x2 - x1;
	
		if (dy < 0) { dy = -dy;  stepy = -1; } else { stepy = 1; }
		if (dx < 0) { dx = -dx;  stepx = -1; } else { stepx = 1; }
		if (dx > dy) {val = 1 / float(dx);}
		else {val = 1 / float(dy);}

		SRM[LOR_ind + y1 * wim + x1] = val;
		SRM[LOR_ind + y2 * wim + x2] = val;
		if (dx > dy) {
			length = (dx - 1) >> 2;
			extras = (dx - 1) & 3;
			incr2 = (dy << 2) - (dx << 1);
			if (incr2 < 0) {
				c = dy << 1;
				incr1 = c << 1;
				d =  incr1 - dx;
				for (i = 0; i < length; i++) {
					x1 = x1 + stepx;
					x2 = x2 - stepx;
					if (d < 0) {                            // Pattern:
						SRM[LOR_ind + y1 * wim + x1] = val; //
						x1 = x1 + stepx;                    // x o o
						SRM[LOR_ind + y1 * wim + x1] = val;
						SRM[LOR_ind + y2 * wim + x2] = val;
						x2 = x2 - stepx;
						SRM[LOR_ind + y2 * wim + x2] = val;
						d += incr1;
					} else {
						if (d < c) {                                 // Pattern:
							SRM[LOR_ind + y1 * wim + x1] = val;      //       o
							x1 = x1 + stepx;                         //   x o
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
							SRM[LOR_ind + y2 * wim + x2] = val;
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
							
						} else {
							y1 = y1 + stepy;                      // Pattern
							SRM[LOR_ind + y1 * wim + x1] = val;   //    o o
							x1 = x1 + stepx;                      //  x
							SRM[LOR_ind + y1 * wim + x1] = val;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
							x2 = x2 - stepx;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d < 0) {
						x1 = x1 + stepx;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
					} else 
	                if (d < c) {
						x1 = x1 + stepx;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
					} else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
	                }
				}
			} else {
			    c = (dy - dx) << 1;
				incr1 = c << 1;
				d =  incr1 + dx;
				for (i = 0; i < length; i++) {
					x1 = x1 + stepx;
					x2 = x2 - stepx;
					if (d > 0) {
						y1 = y1 + stepy;                     // Pattern
						SRM[LOR_ind + y1 * wim + x1] = val;  //      o
						x1 = x1 + stepx;                     //    o
						y1 = y1 + stepy;                     //   x
						SRM[LOR_ind + y1 * wim + x1] = val;
						y2 = y2 - stepy;
						SRM[LOR_ind + y2 * wim + x2] = val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						SRM[LOR_ind + y2 * wim + x2] = val;
						d += incr1;
					} else {
						if (d < c) {
							SRM[LOR_ind + y1 * wim + x1] = val;  // Pattern
							x1 = x1 + stepx;                     //      o
							y1 = y1 + stepy;                     //  x o
							SRM[LOR_ind + y1 * wim + x1] = val;
							SRM[LOR_ind + y2 * wim + x2] = val;
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						} else {
							y1 = y1 + stepy;                    // Pattern
							SRM[LOR_ind + y1 * wim + x1] = val; //    o  o
							x1 = x1 + stepx;                    //  x
							SRM[LOR_ind + y1 * wim + x1] = val;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
							x2 = x2 - stepx;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d > 0) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
					} else 
	                if (d < c) {
						x1 = x1 + stepx;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
					} else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							if (d > c) {
								x2 = x2 - stepx;
								y2 = y2 - stepy;
								SRM[LOR_ind + y2 * wim + x2] = val;
							} else {
								x2 = x2 - stepx;
								SRM[LOR_ind + y2 * wim + x2] = val;
							}
						}
					}
				}
			}
	    } else {
		    length = (dy - 1) >> 2;
			extras = (dy - 1) & 3;
			incr2 = (dx << 2) - (dy << 1);
			if (incr2 < 0) {
				c = dx << 1;
				incr1 = c << 1;
				d =  incr1 - dy;
				for (i = 0; i < length; i++) {
					y1 = y1 + stepy;
					y2 = y2 - stepy;
					if (d < 0) {
						SRM[LOR_ind + y1 * wim + x1] = val;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						SRM[LOR_ind + y2 * wim + x2] = val;
						y2 = y2 - stepy;
						SRM[LOR_ind + y2 * wim + x2] = val;
						d += incr1;
					} else {
						if (d < c) {
							SRM[LOR_ind + y1 * wim + x1] = val;
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
							SRM[LOR_ind + y2 * wim + x2] = val;
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						} else {
							x1 = x1 + stepx;
							SRM[LOR_ind + y1 * wim + x1] = val;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
							x2 = x2 - stepx;
							SRM[LOR_ind + y2 * wim + x2] = val;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d < 0) {
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
					} else 
	                if (d < c) {
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
	                } else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
	                }
				}
	        } else {
				c = (dx - dy) << 1;
				incr1 = c << 1;
				d =  incr1 + dy;
				for (i = 0; i < length; i++) {
					y1 = y1 + stepy;
					y2 = y2 - stepy;
					if (d > 0) {
						x1 = x1 + stepx;
						SRM[LOR_ind + y1 * wim + x1] = val;
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						x2 = x2 - stepx;
						SRM[LOR_ind + y2 * wim + x2] = val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						SRM[LOR_ind + y2 * wim + x2] = val;
						d += incr1;
					} else {
						if (d < c) {
							SRM[LOR_ind + y1 * wim + x1] = val;
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
							SRM[LOR_ind + y2 * wim + x2] = val;
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						} else {
							x1 = x1 + stepx;
							SRM[LOR_ind + y1 * wim + x1] = val;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
							x2 = x2 - stepx;
							SRM[LOR_ind + y2 * wim + x2] = val;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d > 0) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
					} else
	                if (d < c) {
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
	                    if (extras > 2) {
							y2 = y2 - stepy;
							SRM[LOR_ind + y2 * wim + x2] = val;
						}
					} else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						SRM[LOR_ind + y1 * wim + x1] = val;
						if (extras > 1) {
							y1 = y1 + stepy;
							SRM[LOR_ind + y1 * wim + x1] = val;
						}
						if (extras > 2) {
							if (d > c)  {
								x2 = x2 - stepx;
								y2 = y2 - stepy;
								SRM[LOR_ind + y2 * wim + x2] = val;
							} else {
								y2 = y2 - stepy;
								SRM[LOR_ind + y2 * wim + x2] = val;
							}
						}
					}
				}
			}
		}
	}
}

// SRM Raycasting, Compute ray intersection with the 3D SRM
void kernel_pet3D_SRM_raycasting(float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
								 float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
								 int* enable, int nenable, int border, int ROIxy, int ROIz) {
	// Smith's algorithm ray-box AABB intersection
	int i, chk1, chk2;
	float xd, yd, zd, xmin, ymin, zmin, xmax, ymax, zmax;
	float tmin, tmax, tymin, tymax, tzmin, tzmax, buf;
	float xi1, yi1, zi1, xp1, yp1, zp1, xp2, yp2, zp2;
	// define box and ray direction
	xmin = float(border);
	xmax = float(border + ROIxy);
	ymin = float(border);
	ymax = float(border + ROIxy);
	zmin = 0.0f;
	zmax = float(ROIz);
	for (i=0; i<nx1; ++i) {
		xi1 = x1[i];
		yi1 = y1[i];
		zi1 = z1[i];
		xd = x2[i] - xi1;
		yd = y2[i] - yi1;
		zd = z2[i] - zi1;
		tmin = -1e9f;
		tmax = 1e9f;
		// on x
		if (xd != 0.0f) {
			tmin = (xmin - xi1) / xd;
			tmax = (xmax - xi1) / xd;
			if (tmin > tmax) {
				buf = tmin;
				tmin = tmax;
				tmax = buf;
			}
		}
		// on y
		if (yd != 0.0f) {
			tymin = (ymin - yi1) / yd;
			tymax = (ymax - yi1) / yd;
			if (tymin > tymax) {
				buf = tymin;
				tymin = tymax;
				tymax = buf;
			}
			if (tymin > tmin) {tmin = tymin;}
			if (tymax < tmax) {tmax = tymax;}
		}
		// on z
		if (zd != 0.0f) {
			tzmin = (zmin - zi1) / zd;
			tzmax = (zmax - zi1) / zd;
			if (tzmin > tzmax) {
				buf = tzmin;
				tzmin = tzmax;
				tzmax = buf;
			}
			if (tzmin > tmin) {tmin = tzmin;}
			if (tzmax < tmax) {tmax = tzmax;}
		}
		// compute points
		xp1 = xi1 + xd * tmin;
		yp1 = yi1 + yd * tmin;
		zp1 = zi1 + zd * tmin;
		xp2 = xi1 + xd * tmax;
		yp2 = yi1 + yd * tmax;
		zp2 = zi1 + zd * tmax;
		//printf("p1 %f %f %f - p2 %f %f %f\n", xp1, yp1, zp1, xp2, yp2, zp2);
		// check point p1
		chk1 = 0;
		chk2 = 0;
		if (xp1 >= xmin && xp1 <= xmax) {
			if (yp1 >= ymin && yp1 <= ymax) {
				if (zp1 >= zmin && zp1 <= zmax) {
					xp1 -= border;
					yp1 -= border;
					//zp1 -= border;
					if (int(xp1+0.5) == ROIxy) {xp1 = ROIxy-1.0f;}
					if (int(yp1+0.5) == ROIxy) {yp1 = ROIxy-1.0f;}
					if (int(zp1+0.5) == ROIz) {zp1 = ROIz-1.0f;}
					x1[i] = xp1;
					y1[i] = yp1;
					z1[i] = zp1;
					chk1 = 1;
				} else {continue;}
			} else {continue;}
		} else {continue;}
		// check point p2
		if (xp2 >= xmin && xp2 <= xmax) {
			if (yp2 >= ymin && yp2 <= ymax) {
				if (zp2 >= zmin && zp2 <= zmax) {
					xp2 -= border;
					yp2 -= border;
					//zp2 -= border;
					if (int(xp2+0.5) == ROIxy) {xp2 = ROIxy-1.0f;}
					if (int(yp2+0.5) == ROIxy) {yp2 = ROIxy-1.0f;}
					if (int(zp2+0.5) == ROIz) {zp2 = ROIz-1.0f;}
					x2[i] = xp2;
					y2[i] = yp2;
					z2[i] = zp2;
					chk2 = 1;
				} else {continue;}
			} else {continue;}
		} else {continue;}
		if (chk1 && chk2) {
			if (int(xp1) == int(xp2) && int(yp1) == int(yp2) && int(zp1) == int(zp2)) {continue;}
			enable[i] = 1;
		}
	}
}

// Cleanning LORs outside of ROI based on SRM raycasting intersection calculation (return int)
void kernel_pet3D_SRM_clean_LOR_int(int* enable, int ne, float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
									float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
									int* xi1, int nxi1, int* yi1, int nyi1, int* zi1, int nzi1,
									int* xi2, int nxi2, int* yi2, int nyi2, int* zi2, int nzi2) {
	int i, c;
	c = 0;
	for (i=0; i<nx1; ++i) {
		if (enable[i]) {
			xi1[c] = (int)x1[i];
			yi1[c] = (int)y1[i];
			zi1[c] = (int)z1[i];
			xi2[c] = (int)x2[i];
			yi2[c] = (int)y2[i];
			zi2[c] = (int)z2[i];
			++c;
		}
	}
}

// Cleanning LORs outside of ROI based on SRM raycasting intersection calculation (return float)
void kernel_pet3D_SRM_clean_LOR_float(int* enable, int ne, float* x1, int nx1, float* y1, int ny1, float* z1, int nz1,
									  float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
									  float* xf1, int nxi1, float* yf1, int nyi1, float* zf1, int nzi1,
									  float* xf2, int nxi2, float* yf2, int nyi2, float* zf2, int nzi2) {
	int i, c;
	c = 0;
	for (i=0; i<nx1; ++i) {
		if (enable[i]) {
			xf1[c] = x1[i];
			yf1[c] = y1[i];
			zf1[c] = z1[i];
			xf2[c] = x2[i];
			yf2[c] = y2[i];
			zf2[c] = z2[i];
			++c;
		}
	}
}


// Raytrace SRM matrix with DDA algorithm in ELL sparse matrix format
void kernel_pet3D_SRM_ELL_DDA(float* vals, int niv, int njv, int* cols, int nic, int njc,
							  unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1, unsigned short int* Z1, int nz1,
							  unsigned short int* X2, int nx2, unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2, int wim) {
	int length, lengthy, lengthz, i, n;
	float flength, val;
	float x, y, z, lx, ly, lz;
	float xinc, yinc, zinc;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int LOR_ind;
	int step;
	val = 1.0f;
	step = wim*wim;
	
	for (i=0; i< nx1; ++i) {
		LOR_ind = i * njv;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		z1 = Z1[i];
		z2 = Z2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		diffz = z2-z1;
		lx = abs(diffx);
		ly = abs(diffy);
		lz = abs(diffz);
		length = ly;
		if (lx > length) {length = lx;}
		if (lz > length) {length = lz;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		zinc = diffz / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
		z = z1 + 0.5;
		for (n=0; n<=length; ++n) {
			vals[LOR_ind + n] = val;
			cols[LOR_ind + n] = (int)z * step + (int)y * wim + (int)x;
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
		cols[LOR_ind + n] = -1; // eof
	}
}

// Compute the first image with DDA algorithm
void kernel_pet3D_IM_SRM_DDA( unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1, unsigned short int* Z1, int nz1,
							  unsigned short int* X2, int nx2, unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
							  float* im, int nim, int wim) {
	int length, lengthy, lengthz, i, n;
	float flength, val;
	float x, y, z, lx, ly, lz;
	float xinc, yinc, zinc;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int step;
	val = 1.0f;
	step = wim*wim;
	
	for (i=0; i< nx1; ++i) {
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		z1 = Z1[i];
		z2 = Z2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		diffz = z2-z1;
		lx = abs(diffx);
		ly = abs(diffy);
		lz = abs(diffz);
		length = ly;
		if (lx > length) {length = lx;}
		if (lz > length) {length = lz;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		zinc = diffz / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
		z = z1 + 0.5;
		for (n=0; n<=length; ++n) {
			im[(int)z * step + (int)y * wim + (int)x] += val;
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
	}
}

// Update image online, SRM is build with DDA's Line Algorithm, store in ELL format and update with LM-OSEM
void kernel_pet3D_IM_SRM_ELL_DDA_iter(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1, unsigned short int* Z1, int nz1,
									  unsigned short int* X2, int nx2, unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
									  float* im, int nim, float* F, int nf, int wim, int ndata) {
	int length, lengthy, lengthz, i, j, n;
	float flength, val;
	float x, y, z, lx, ly, lz;
	float xinc, yinc, zinc;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int step;
	val = 1.0f;
	step = wim*wim;

	// alloc mem
	float* vals = (float*)malloc(nx1 * ndata * sizeof(float));
	int* cols = (int*)malloc(nx1 * ndata * sizeof(int));
	float* Q = (float*)calloc(nx1, sizeof(float));
	int LOR_ind;
	// to compute F
	int vcol;
	float buf, sum, Qi;

	for (i=0; i< nx1; ++i) {
		LOR_ind = i * ndata;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		z1 = Z1[i];
		z2 = Z2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		diffz = z2-z1;
		lx = abs(diffx);
		ly = abs(diffy);
		lz = abs(diffz);
		length = ly;
		if (lx > length) {length = lx;}
		if (lz > length) {length = lz;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		zinc = diffz / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
		z = z1 + 0.5;
		for (n=0; n<=length; ++n) {
			vals[LOR_ind + n] = val;
			cols[LOR_ind + n] = (int)z * step + (int)y * wim + (int)x;
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
		// eof
		vals[LOR_ind + n] = -1;
		cols[LOR_ind + n] = -1;
	}

	// Sparse matrix operation Q = SRM * im
	for (i=0; i<nx1; ++i) {
		LOR_ind = i * ndata;
		vcol = cols[LOR_ind];
		j = 0;
		sum = 0.0f;
		while (vcol != -1) {
			sum += (vals[LOR_ind+j] * im[vcol]);
			++j;
			vcol = cols[LOR_ind+j];
		}
		Q[i] = sum;
	}
	// Sparse matrix operation F = SRM^T / Q
	for (i=0; i<nx1; ++i) {
		LOR_ind = i * ndata;
		vcol = cols[LOR_ind];
		j = 0;
		Qi = Q[i];
		if (Qi==0.0f) {continue;}
		while (vcol != -1) {
			F[vcol] += (vals[LOR_ind+j] / Qi);
			++j;
			vcol = cols[LOR_ind+j];
		}
	}

	free(vals);
	free(cols);
	free(Q);

	
}


// Update image online, SRM is build with DDA's Line Algorithm, store in ELL format and update with LM-OSEM
void kernel_pet3D_IM_SRM_ELL_DDA_iter_vec(unsigned short int* X1, int nx1, unsigned short int* Y1, int ny1, unsigned short int* Z1, int nz1,
									  unsigned short int* X2, int nx2, unsigned short int* Y2, int ny2, unsigned short int* Z2, int nz2,
									  float* im, int nim, float* F, int nf, int wim, int ndata) {
	int length, lengthy, lengthz, i, j, n;
	float flength, val;
	float x, y, z, lx, ly, lz;
	float xinc, yinc, zinc;
	int x1, y1, z1, x2, y2, z2, diffx, diffy, diffz;
	int step;
	val = 1.0f;
	step = wim*wim;

	// alloc mem
	float* vals = (float*)malloc(ndata * sizeof(float));
	int* cols = (int*)malloc(ndata * sizeof(int));
	int LOR_ind;
	// to compute F
	int vcol;
	float buf, sum, Qi;

	for (i=0; i< nx1; ++i) {
		Qi = 0.0f;
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		z1 = Z1[i];
		z2 = Z2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		diffz = z2-z1;
		lx = abs(diffx);
		ly = abs(diffy);
		lz = abs(diffz);
		length = ly;
		if (lx > length) {length = lx;}
		if (lz > length) {length = lz;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		zinc = diffz / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
		z = z1 + 0.5;
		for (n=0; n<=length; ++n) {
			vals[n] = val;
			vcol = (int)z * step + (int)y * wim + (int)x;
			cols[n] = vcol;
			Qi += (val * im[vcol]);
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
		// eof
		vals[n] = -1;
		cols[n] = -1;
		// compute F
		if (Qi==0.0f) {continue;}
		vcol = cols[0];
		j = 0;
		while (vcol != -1) {
			F[vcol] += (vals[j] / Qi);
			++j;
			vcol = cols[j];
		}
	}
	free(vals);
	free(cols);
}

// Compute first image ionline by Siddon's Line Algorithm
void kernel_pet3D_IM_SRM_SIDDON(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
								float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2, float* im, int nim, int wim, int dim) {
	int n;
	float tx, ty, tz, px, qx, py, qy, pz, qz;
	int ei, ej, ek, u, v, w, i, j, k, oldi, oldj, oldk;
	int stepi, stepj, stepk;
	float divx, divy, divz, runx, runy, runz, oldv, newv, val, valmax;
	float axstart, aystart, azstart, astart, pq, stepx, stepy, stepz, startl, initl;
	int wim2 = wim*wim;

	// random seed
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		px = X2[n];
		py = Y2[n];
		pz = Z2[n];
		qx = X1[n];
		qy = Y1[n];
		qz = Z1[n];
		initl = (float)rand() / (float)RAND_MAX;
		//initl = initl * 0.6 + 0.2; // rnd number between 0.2 to 0.8
		initl = initl * 0.4 + 0.1;
		tx = (px-qx) * initl + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * initl + qy;
		tz = (pz-qz) * initl + qz;
		ei = int(tx);
		ej = int(ty);
		ek = int(tz);
		if (qx-tx>0) {
			u=ei+1;
			stepi=1;
		}
		if (qx-tx<0) {
			u=ei;
			stepi=-1;
		}
		if (qx-tx==0) {
			u=ei;
			stepi=0;
		}
		if (qy-ty>0) {
			v=ej+1;
			stepj=1;
		}
		if (qy-ty<0) {
			v=ej;
			stepj=-1;
		}
		if (qy-ty==0) {
			v=ej;
			stepj=0;
		}
		if (qz-tz>0) {
			w=ek+1;
			stepk=1;
		}
		if (qz-tz<0) {
			w=ek;
			stepk=-1;
		}
		if (qz-tz==0) {
			w=ej;
			stepk=0;
		}
		
		if (qx==px) {divx=1.0;}
		else {divx = float(qx-px);}
		if (qy==py) {divy=1.0;}
		else {divy = float(qy-py);}
		if (qz==pz) {divz=1.0;}
		else {divz = float(qz-pz);}
		axstart = (u-px) / divx;
		aystart = (v-py) / divy;
		azstart = (w-pz) / divz;
		astart = aystart;
		if (axstart > aystart) {astart = axstart;}
		if (azstart > astart) {astart = azstart;}
		pq = sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py)+(qz-pz)*(qz-pz));
		stepx = fabs(pq / divx);
		stepy = fabs(pq / divy);
		stepz = fabs(pq / divz);
		startl = astart * pq;
		valmax = stepx;
		if (stepy < valmax) {valmax = stepy;}
		if (stepz < valmax) {valmax = stepz;}
		valmax = valmax + valmax*0.01f;

		// first half-ray
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		i = ei;
		j = ej;
		k = ek;
		if (runx == startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy == startl) {
			j += stepj;
			runy += stepy;
		}
		if (runz == startl) {
			k += stepk;
			runz += stepz;
		}
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<dim) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {im[k * wim2 + j * wim + i] += val;}
			oldv = newv;
			oldi = i;
			oldj = j;
			oldk = k;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
			if (runz == newv) {
				k += stepk;
				runz += stepz;
			}
		}
		// second half-ray
		if (px-tx>0) {stepi=1;}
		if (px-tx<0) {stepi=-1;}
		if (py-ty>0) {stepj=1;}
		if (py-ty<0) {stepj=-1;}
		if (pz-tz>0) {stepk=1;}
		if (pz-tz<0) {stepk=-1;}
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		i = ei;
		j = ej;
		k = ek;
		if (runx==startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy==startl) {
			j += stepj;
			runy += stepy;
		}
		if (runz==startl) {
			k += stepk;
			runz += stepz;
		}
		im[ek * wim2 + ej * wim + ei] += 0.707f; //val;
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<dim) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {im[k * wim2 + j * wim + i] += val;}
			oldv = newv;
			oldi = i;
			oldj = j;
			oldk = k;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
			if (runz == newv) {
				k += stepk;
				runz += stepz;
			}
		}
	}
}

// Update image online, SRM is build with Siddon's Line Algorithm, and update with LM-OSEM
void kernel_pet3D_IM_SRM_SIDDON_iter(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
									 float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2,
									 float* im, int nim, float* F, int nf, int wim) {
	int n;
	float tx, ty, tz, px, qx, py, qy, pz, qz;
	int ei, ej, ek, u, v, w, i, j, k, oldi, oldj, oldk;
	int stepi, stepj, stepk;
	float divx, divy, divz, runx, runy, runz, oldv, newv, val, valmax;
	float axstart, aystart, azstart, astart, pq, stepx, stepy, stepz, startl, initl;
	int wim2 = wim*wim;
	double Qi;
	float* SRM = (float*)malloc(nim * sizeof(float));

	// random seed
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		//printf("%i\n", n);
		// init SRM and Qi
		//for (i=0; i<nim; ++i) {SRM[i] = 0.0f;}
		memset(SRM, 0, nim*sizeof(float));
		Qi = 0.0f;
		// draw the line
		px = X2[n];
		py = Y2[n];
		pz = Z2[n];
		qx = X1[n];
		qy = Y1[n];
		qz = Z1[n];
		initl = (float)rand() / (float)RAND_MAX;
		//initl = initl * 0.6 + 0.2; // rnd number between 0.2 to 0.8
		initl = initl * 0.4 + 0.1;
		tx = (px-qx) * initl + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * initl + qy;
		tz = (pz-qz) * initl + qz;
		ei = int(tx);
		ej = int(ty);
		ek = int(tz);
		if (qx-tx>0) {
			u=ei+1;
			stepi=1;
		}
		if (qx-tx<0) {
			u=ei;
			stepi=-1;
		}
		if (qx-tx==0) {
			u=ei;
			stepi=0;
		}
		if (qy-ty>0) {
			v=ej+1;
			stepj=1;
		}
		if (qy-ty<0) {
			v=ej;
			stepj=-1;
		}
		if (qy-ty==0) {
			v=ej;
			stepj=0;
		}
		if (qz-tz>0) {
			w=ek+1;
			stepk=1;
		}
		if (qz-tz<0) {
			w=ek;
			stepk=-1;
		}
		if (qz-tz==0) {
			w=ej;
			stepk=0;
		}
		
		if (qx==px) {divx=1.0;}
		else {divx = float(qx-px);}
		if (qy==py) {divy=1.0;}
		else {divy = float(qy-py);}
		if (qz==pz) {divz=1.0;}
		else {divz = float(qz-pz);}
		axstart = (u-px) / divx;
		aystart = (v-py) / divy;
		azstart = (w-pz) / divz;
		astart = aystart;
		if (axstart > aystart) {astart = axstart;}
		if (azstart > astart) {astart = azstart;}
		pq = sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py)+(qz-pz)*(qz-pz));
		stepx = fabs(pq / divx);
		stepy = fabs(pq / divy);
		stepz = fabs(pq / divz);
		startl = astart * pq;
		valmax = stepx;
		if (stepy < valmax) {valmax = stepy;}
		if (stepz < valmax) {valmax = stepz;}
		valmax = valmax + valmax*0.01f;

		// first half-ray
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		i = ei;
		j = ej;
		k = ek;
		if (runx == startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy == startl) {
			j += stepj;
			runy += stepy;
		}
		if (runz == startl) {
			k += stepk;
			runz += stepz;
		}
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<wim) {
			
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {SRM[k * wim2 + j * wim + i] += val;}
			oldv = newv;
			oldi = i;
			oldj = j;
			oldk = k;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
			if (runz == newv) {
				k += stepk;
				runz += stepz;
			}
		}
		// second half-ray
		if (px-tx>0) {stepi=1;}
		if (px-tx<0) {stepi=-1;}
		if (py-ty>0) {stepj=1;}
		if (py-ty<0) {stepj=-1;}
		if (pz-tz>0) {stepk=1;}
		if (pz-tz<0) {stepk=-1;}
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		i = ei;
		j = ej;
		k = ek;
		if (runx==startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy==startl) {
			j += stepj;
			runy += stepy;
		}
		if (runz==startl) {
			k += stepk;
			runz += stepz;
		}
		SRM[ek * wim2 + ej * wim + ei] += 0.707f; //val;
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<wim) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {SRM[k * wim2 + j * wim + i] += val;}
			oldv = newv;
			oldi = i;
			oldj = j;
			oldk = k;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
			if (runz == newv) {
				k += stepk;
				runz += stepz;
			}
		}
		// first compute Qi
		for (i=0; i<nim; ++i) {Qi += (SRM[i] * im[i]);}
		if (Qi == 0.0f) {continue;}
		// accumulate to F
		for (i=0; i<nim; ++i) {
			if (im[i] != 0.0f) {
				F[i] += (SRM[i] / Qi);
			}
		}
		
	} // LORs loop
	free(SRM);
	
}

// Update image online, SRM is build with Siddon's Line Algorithm in COO format, and update with LM-OSEM
void kernel_pet3D_IM_SRM_COO_ON_SIDDON_iter(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
											float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2,
											float* im, int nim, float* F, int nf, int wim, int dim) {
	int n, ct;
	float tx, ty, tz, px, qx, py, qy, pz, qz;
	int ei, ej, ek, u, v, w, i, j, k, oldi, oldj, oldk;
	int stepi, stepj, stepk;
	float divx, divy, divz, runx, runy, runz, oldv, newv, val, valmax;
	float axstart, aystart, azstart, astart, pq, stepx, stepy, stepz, startl, initl;
	int wim2 = wim*wim;
	double Qi;
	float* vals = NULL;
	int* cols = NULL;

	// random seed
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		float* vals = NULL;
		int* cols = NULL;
		Qi = 0.0f;
		ct = 0;
		// draw the line
		px = X2[n];
		py = Y2[n];
		pz = Z2[n];
		qx = X1[n];
		qy = Y1[n];
		qz = Z1[n];
		initl = (float)rand() / (float)RAND_MAX;
		//initl = initl * 0.6 + 0.2; // rnd number between 0.2 to 0.8
		initl = initl * 0.4 + 0.1;
		tx = (px-qx) * initl + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * initl + qy;
		tz = (pz-qz) * initl + qz;
		ei = int(tx);
		ej = int(ty);
		ek = int(tz);
		if (qx-tx>0) {
			u=ei+1;
			stepi=1;
		}
		if (qx-tx<0) {
			u=ei;
			stepi=-1;
		}
		if (qx-tx==0) {
			u=ei;
			stepi=0;
		}
		if (qy-ty>0) {
			v=ej+1;
			stepj=1;
		}
		if (qy-ty<0) {
			v=ej;
			stepj=-1;
		}
		if (qy-ty==0) {
			v=ej;
			stepj=0;
		}
		if (qz-tz>0) {
			w=ek+1;
			stepk=1;
		}
		if (qz-tz<0) {
			w=ek;
			stepk=-1;
		}
		if (qz-tz==0) {
			w=ej;
			stepk=0;
		}
		
		if (qx==px) {divx=1.0;}
		else {divx = float(qx-px);}
		if (qy==py) {divy=1.0;}
		else {divy = float(qy-py);}
		if (qz==pz) {divz=1.0;}
		else {divz = float(qz-pz);}
		axstart = (u-px) / divx;
		aystart = (v-py) / divy;
		azstart = (w-pz) / divz;
		astart = aystart;
		if (axstart > aystart) {astart = axstart;}
		if (azstart > astart) {astart = azstart;}
		pq = sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py)+(qz-pz)*(qz-pz));
		stepx = fabs(pq / divx);
		stepy = fabs(pq / divy);
		stepz = fabs(pq / divz);
		startl = astart * pq;
		valmax = stepx;
		if (stepy < valmax) {valmax = stepy;}
		if (stepz < valmax) {valmax = stepz;}
		valmax = valmax + valmax*0.01f;

		// first half-ray
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		i = ei;
		j = ej;
		k = ek;
		if (runx == startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy == startl) {
			j += stepj;
			runy += stepy;
		}
		if (runz == startl) {
			k += stepk;
			runz += stepz;
		}
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<dim) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {
				++ct;
				vals = (float*)realloc(vals, ct*sizeof(float));
				cols = (int*)realloc(cols, ct*sizeof(int));
				vals[ct-1] = val;
				cols[ct-1] = k * wim2 + j * wim + i;
			}
			oldv = newv;
			oldi = i;
			oldj = j;
			oldk = k;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
			if (runz == newv) {
				k += stepk;
				runz += stepz;
			}
		}
		// second half-ray
		if (px-tx>0) {stepi=1;}
		if (px-tx<0) {stepi=-1;}
		if (py-ty>0) {stepj=1;}
		if (py-ty<0) {stepj=-1;}
		if (pz-tz>0) {stepk=1;}
		if (pz-tz<0) {stepk=-1;}
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		i = ei;
		j = ej;
		k = ek;
		if (runx==startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy==startl) {
			j += stepj;
			runy += stepy;
		}
		if (runz==startl) {
			k += stepk;
			runz += stepz;
		}
		++ct;
		vals = (float*)realloc(vals, ct*sizeof(float));
		cols = (int*)realloc(cols, ct*sizeof(int));
		vals[ct-1] = 0.707f;
		cols[ct-1] = ek * wim2 + ej * wim + ei;
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<dim) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {
				++ct;
				vals = (float*)realloc(vals, ct*sizeof(float));
				cols = (int*)realloc(cols, ct*sizeof(int));
				vals[ct-1] = val;
				cols[ct-1] = k * wim2 + j * wim + i;
			}
			oldv = newv;
			oldi = i;
			oldj = j;
			oldk = k;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
			if (runz == newv) {
				k += stepk;
				runz += stepz;
			}
		}
		// first compute Qi
		for (i=0; i<ct; ++i) {Qi += (vals[i] * im[cols[i]]);}
		if (Qi == 0.0f) {continue;}
		// accumulate to F
		for(i=0; i<ct; ++i) {
			if (im[cols[i]] != 0.0f) {
				F[cols[i]] += (vals[i] / Qi);
			}

		}
		free(vals);
		free(cols);
		
	} // LORs loop
	
}

// Compute first image online by Siddon's Line Algorithm, and store SRM matrix to the harddrive with COO format
void kernel_pet3D_IM_SRM_COO_SIDDON(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
									float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2, float* im, int nim, int wim, int isub) {
	int n;
	float tx, ty, tz, px, qx, py, qy, pz, qz;
	int ei, ej, ek, u, v, w, i, j, k, oldi, oldj, oldk;
	int stepi, stepj, stepk;
	float divx, divy, divz, runx, runy, runz, oldv, newv, val, valmax;
	float axstart, aystart, azstart, astart, pq, stepx, stepy, stepz, startl, initl;
	int wim2 = wim*wim;
	int col, ct;

	// init file
	FILE * pfile_vals;
	FILE * pfile_rows;
	FILE * pfile_cols;
	char namevals [20];
	char namecols [20];
	char namerows [20];
	sprintf(namevals, "SRMvals_%i.coo", isub);
	sprintf(namecols, "SRMcols_%i.coo", isub);
	sprintf(namerows, "SRMrows_%i.coo", isub);
	pfile_vals = fopen(namevals, "wb");
	pfile_cols = fopen(namecols, "wb");
	pfile_rows = fopen(namerows, "wb");

	// random seed
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		//printf("%i %f %f %f - %f %f %f\n", n, px, py, pz, qx, qy, qz);
		ct = 0;
		px = X2[n];
		py = Y2[n];
		pz = Z2[n];
		qx = X1[n];
		qy = Y1[n];
		qz = Z1[n];
		initl = (float)rand() / (float)RAND_MAX;
		initl = initl * 0.6 + 0.2; // rnd number between 0.2 to 0.8
		tx = (px-qx) * initl + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * initl + qy;
		tz = (pz-qz) * initl + qz;
		ei = int(tx);
		ej = int(ty);
		ek = int(tz);
		if (qx-tx>0) {
			u=ei+1;
			stepi=1;
		}
		if (qx-tx<0) {
			u=ei;
			stepi=-1;
		}
		if (qx-tx==0) {
			u=ei;
			stepi=0;
		}
		if (qy-ty>0) {
			v=ej+1;
			stepj=1;
		}
		if (qy-ty<0) {
			v=ej;
			stepj=-1;
		}
		if (qy-ty==0) {
			v=ej;
			stepj=0;
		}
		if (qz-tz>0) {
			w=ek+1;
			stepk=1;
		}
		if (qz-tz<0) {
			w=ek;
			stepk=-1;
		}
		if (qz-tz==0) {
			w=ej;
			stepk=0;
		}
		
		if (qx==px) {divx=1.0;}
		else {divx = float(qx-px);}
		if (qy==py) {divy=1.0;}
		else {divy = float(qy-py);}
		if (qz==pz) {divz=1.0;}
		else {divz = float(qz-pz);}
		axstart = (u-px) / divx;
		aystart = (v-py) / divy;
		azstart = (w-pz) / divz;
		astart = aystart;
		if (axstart > aystart) {astart = axstart;}
		if (azstart > astart) {astart = azstart;}
		pq = sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py)+(qz-pz)*(qz-pz));
		stepx = fabs(pq / divx);
		stepy = fabs(pq / divy);
		stepz = fabs(pq / divz);
		startl = astart * pq;
		valmax = stepx;
		if (stepy < valmax) {valmax = stepy;}
		if (stepz < valmax) {valmax = stepz;}
		valmax = valmax + valmax*0.01f;

		// first half-ray
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		i = ei;
		j = ej;
		k = ek;
		if (runx == startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy == startl) {
			j += stepj;
			runy += stepy;
		}
		if (runz == startl) {
			k += stepk;
			runz += stepz;
		}
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<wim) {
			
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {
				col = k * wim2 + j * wim + i;
				im[col] += val;
				fwrite(&val, sizeof(float), 1, pfile_vals);
				fwrite(&col, sizeof(int), 1, pfile_cols);
				fwrite(&n, sizeof(int), 1, pfile_rows);
				++ct;
			}
			oldv = newv;
			oldi = i;
			oldj = j;
			oldk = k;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
			if (runz == newv) {
				k += stepk;
				runz += stepz;
			}
		}
		// second half-ray
		if (px-tx>0) {stepi=1;}
		if (px-tx<0) {stepi=-1;}
		if (py-ty>0) {stepj=1;}
		if (py-ty<0) {stepj=-1;}
		if (pz-tz>0) {stepk=1;}
		if (pz-tz<0) {stepk=-1;}
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		i = ei;
		j = ej;
		k = ek;
		if (runx==startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy==startl) {
			j += stepj;
			runy += stepy;
		}
		if (runz==startl) {
			k += stepk;
			runz += stepz;
		}
		col = ek * wim2 + ej * wim + ei;
		val = 0.707f;
		im[col] += val;
		fwrite(&val, sizeof(float), 1, pfile_vals);
		fwrite(&col, sizeof(int), 1, pfile_cols);
		fwrite(&n, sizeof(int), 1, pfile_rows);
		++ct;
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<wim) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {
				col = k * wim2 + j * wim + i;
				im[col] += val;
				fwrite(&val, sizeof(float), 1, pfile_vals);
				fwrite(&col, sizeof(int), 1, pfile_cols);
				fwrite(&n, sizeof(int), 1, pfile_rows);
				++ct;
			}
			oldv = newv;
			oldi = i;
			oldj = j;
			oldk = k;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
			if (runz == newv) {
				k += stepk;
				runz += stepz;
			}
		}
	}
	// close files
	fclose(pfile_vals);
	fclose(pfile_cols);
	fclose(pfile_rows);
}

// Update image online, SRM is read from the hard-drive and update with LM-OSEM
void kernel_pet3D_IM_SRM_COO_SIDDON_iter_vec(float* im, int nim, float* F, int nf, int N, int isub) {
	// open files
	FILE * pfile_vals;
	FILE * pfile_rows;
	FILE * pfile_cols;
	char namevals [20];
	char namecols [20];
	char namerows [20];
	sprintf(namevals, "SRMvals_%i.coo", isub);
	sprintf(namecols, "SRMcols_%i.coo", isub);
	sprintf(namerows, "SRMrows_%i.coo", isub);
	pfile_vals = fopen(namevals, "rb");
	pfile_cols = fopen(namecols, "rb");
	pfile_rows = fopen(namerows, "rb");

	// init
	//float* SRM = (float*)malloc(nim * sizeof(float));
	int* Ni = (int*)calloc(N, sizeof(int));
	float Qi, ival;
	int i, n, icol;
	// compute number of elements per rows
	int nbele;
	fseek(pfile_rows, 0, SEEK_END);
	nbele = ftell(pfile_rows);
	rewind(pfile_rows);
	nbele /= sizeof(float);
	int irows;
	for (i=0; i<nbele; ++i) {
		fread(&irows, 1, sizeof(int), pfile_rows);
		Ni[irows] += 1;
	}
	// create a static memory
	int max = 0;
	for (i=0; i<N; ++i) {
		if (Ni[i]>max) {max=Ni[i];}
	}
	float* vals = (float*)malloc(max * sizeof(float));
	int* cols = (int*)malloc(max * sizeof(int));
	
	// read SRM
	for (n=0; n<N; ++n) {
		
		// init SRM and Qi
		Qi = 0.0f;
		for (i=0; i<Ni[n]; ++i) {
			fread(&icol, 1, sizeof(int), pfile_cols);
			fread(&ival, 1, sizeof(float), pfile_vals);
			vals[i] = ival;
			cols[i] = icol;
			Qi += (ival * im[icol]);
		}
		if (Qi == 0.0f) {continue;}
		// accumulate to F
		for (i=0; i<Ni[n]; ++i) {
			F[cols[i]] += (vals[i] / Qi);
		}

	}
	// close files
	fclose(pfile_vals);
	fclose(pfile_cols);
	fclose(pfile_rows);
	free(Ni);
	free(vals);
	free(cols);
	
}

// Update image online, SRM is read from the hard-drive and update with LM-OSEM
void kernel_pet3D_IM_SRM_COO_SIDDON_iter_mat(float* vals, int nvals, int* cols, int ncols, int* rows, int nrows, float* im, int nim, float* F, int nf, int N, int isub) {
	int i, j, ind;
	float buf;
	float* Q = (float*)malloc(N * sizeof(float));

	// init Q and F
	for (i=0; i<N; ++i) {Q[i] = 0.0f;}
	
	// Sparse matrix multiplication Q = SRM * im
	for (i=0; i<nvals; ++i) {
		Q[rows[i]] += (vals[i] * im[cols[i]]);
	}
	// Sparse matrix operation F = SRM^T / Q
	for (i=0; i<nvals; ++i) {
		if (Q[rows[i]] == 0.0f) {continue;}
		F[cols[i]] += (vals[i] / Q[rows[i]]);
	}
	/*
	// update pixel
	for (j=0; j<npix; ++j) {
		buf = im[j];
		if (buf != 0) {
			im[j] = buf / S[j] * F[j];
		}
	}
	*/
	free(Q);
}

// Update image online, SRM is build with Siddon's Line Algorithm, store in ELL format and update with LM-OSEM
void kernel_pet3D_IM_SRM_ELL_SIDDON_iter(float* X1, int nx1, float* Y1, int ny1, float* Z1, int nz1,
										 float* X2, int nx2, float* Y2, int ny2, float* Z2, int nz2,
										 float* im, int nim, float* F, int nf, int wim, int ndata) {
	int n;
	float tx, ty, tz, px, qx, py, qy, pz, qz;
	int ei, ej, ek, u, v, w, i, j, k, oldi, oldj, oldk;
	int stepi, stepj, stepk;
	float divx, divy, divz, runx, runy, runz, oldv, newv, val, valmax;
	float axstart, aystart, azstart, astart, pq, stepx, stepy, stepz, startl, initl;
	int wim2 = wim*wim;

	// alloc mem
	float* vals = (float*)malloc(nx1 * ndata * sizeof(float));
	int* cols = (int*)malloc(nx1 * ndata * sizeof(int));
	float* Q = (float*)calloc(nx1, sizeof(float));
	int ct, LOR_ind;
	// to compute F
	int vcol;
	float buf, sum, Qi;

	// random seed
	srand(time(NULL));
	for (n=0; n<nx1; ++n) {
		LOR_ind = n * ndata;
		ct = 0;
		// draw the line
		px = X2[n];
		py = Y2[n];
		pz = Z2[n];
		qx = X1[n];
		qy = Y1[n];
		qz = Z1[n];
		initl = (float)rand() / (float)RAND_MAX;
		initl = initl * 0.6 + 0.2; // rnd number between 0.2 to 0.8
		tx = (px-qx) * initl + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * initl + qy;
		tz = (pz-qz) * initl + qz;
		ei = int(tx);
		ej = int(ty);
		ek = int(tz);
		if (qx-tx>0) {
			u=ei+1;
			stepi=1;
		}
		if (qx-tx<0) {
			u=ei;
			stepi=-1;
		}
		if (qx-tx==0) {
			u=ei;
			stepi=0;
		}
		if (qy-ty>0) {
			v=ej+1;
			stepj=1;
		}
		if (qy-ty<0) {
			v=ej;
			stepj=-1;
		}
		if (qy-ty==0) {
			v=ej;
			stepj=0;
		}
		if (qz-tz>0) {
			w=ek+1;
			stepk=1;
		}
		if (qz-tz<0) {
			w=ek;
			stepk=-1;
		}
		if (qz-tz==0) {
			w=ej;
			stepk=0;
		}
		
		if (qx==px) {divx=1.0;}
		else {divx = float(qx-px);}
		if (qy==py) {divy=1.0;}
		else {divy = float(qy-py);}
		if (qz==pz) {divz=1.0;}
		else {divz = float(qz-pz);}
		axstart = (u-px) / divx;
		aystart = (v-py) / divy;
		azstart = (w-pz) / divz;
		astart = aystart;
		if (axstart > aystart) {astart = axstart;}
		if (azstart > astart) {astart = azstart;}
		pq = sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py)+(qz-pz)*(qz-pz));
		stepx = fabs(pq / divx);
		stepy = fabs(pq / divy);
		stepz = fabs(pq / divz);
		startl = astart * pq;
		valmax = stepx;
		if (stepy < valmax) {valmax = stepy;}
		if (stepz < valmax) {valmax = stepz;}
		valmax = valmax + valmax*0.01f;

		// first half-ray
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		i = ei;
		j = ej;
		k = ek;
		if (runx == startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy == startl) {
			j += stepj;
			runy += stepy;
		}
		if (runz == startl) {
			k += stepk;
			runz += stepz;
		}
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<wim) {
			
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {
				vals[LOR_ind + ct] = val;
				cols[LOR_ind + ct] = k * wim2 + j * wim + i;
				++ct;
			}
			oldv = newv;
			oldi = i;
			oldj = j;
			oldk = k;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
			if (runz == newv) {
				k += stepk;
				runz += stepz;
			}
		}
		// second half-ray
		if (px-tx>0) {stepi=1;}
		if (px-tx<0) {stepi=-1;}
		if (py-ty>0) {stepj=1;}
		if (py-ty<0) {stepj=-1;}
		if (pz-tz>0) {stepk=1;}
		if (pz-tz<0) {stepk=-1;}
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		i = ei;
		j = ej;
		k = ek;
		if (runx==startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy==startl) {
			j += stepj;
			runy += stepy;
		}
		if (runz==startl) {
			k += stepk;
			runz += stepz;
		}
		vals[LOR_ind + ct] = 0.707f;
		cols[LOR_ind + ct] = ek * wim2 + ej * wim + ei;
		++ct;
		oldv = startl;
		oldi = -1;
		oldj = -1;
		oldk = -1;
		while (i>=0 && j>=0 && k>=0 && i<wim && j<wim && k<wim) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			if (runz < newv) {newv = runz;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			if (oldi != i || oldj != j || oldk != k) {
				vals[LOR_ind + ct] = val;
				cols[LOR_ind + ct] = k * wim2 + j * wim + i;
				++ct;
			}
			oldv = newv;
			oldi = i;
			oldj = j;
			oldk = k;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
			if (runz == newv) {
				k += stepk;
				runz += stepz;
			}
		}
		// eof		
		vals[LOR_ind + ct] = -1;
		cols[LOR_ind + ct] = -1;
		//printf("ct %i\n", ct);
	} // LORs loop

	// Sparse matrix operation Q = SRM * im
	for (i=0; i<nx1; ++i) {
		LOR_ind = i * ndata;
		vcol = cols[LOR_ind];
		j = 0;
		sum = 0.0f;
		while (vcol != -1) {
			sum += (vals[LOR_ind+j] * im[vcol]);
			++j;
			vcol = cols[LOR_ind+j];
		}
		Q[i] = sum;
	}
	// Sparse matrix operation F = SRM^T / Q
	for (i=0; i<nx1; ++i) {
		LOR_ind = i * ndata;
		vcol = cols[LOR_ind];
		j = 0;
		Qi = Q[i];
		if (Qi==0.0f) {continue;}
		while (vcol != -1) {
			F[vcol] += (vals[LOR_ind+j] / Qi);
			++j;
			vcol = cols[LOR_ind+j];
		}
	}
	free(vals);
	free(cols);
	free(Q);
}


/********************************************************************************
 * GENERAL      volume rendering
 ********************************************************************************/
// helper function to rendering volume
void kernel_draw_voxels(int* posxyz, int npos, float* val, int nval, float* valthr, int nthr, float gamma, float thres){
	int ind, n, x, y, z;
	float r, g, b, l;
	for (n=0; n<nthr; ++n) {
		l = valthr[n];
		if (l <= thres) {continue;}
		ind = 3 * n;
		x = posxyz[ind];
		y = posxyz[ind+1];
		z = posxyz[ind+2];
		r = val[ind];
		g = val[ind+1];
		b = val[ind+2];
		l *= gamma;
		glColor4f(r, g, b, l);
		// face 0
		glBegin(GL_QUADS);
		glNormal3f(-1, 0, 0);
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y+1.0, z); // 2
		glVertex3f(x, y+1.0, z+1.0); // 3
		glVertex3f(x, y, z+1.0); // 4
		glEnd();
		// face 1
		glBegin(GL_QUADS);
		glNormal3f(0, 1, 0);
		glVertex3f(x, y+1, z+1); // 3
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y+1, z+1); // 7
		glEnd();
		// face 2 
		glBegin(GL_QUADS);
		glNormal3f(1, 0, 0);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y, z+1); // 4
		glEnd();
		// face 3
		glBegin(GL_QUADS);
		glNormal3f(0, -1, 0);
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y, z+1); // 0
		glEnd();
		// face 4
		glBegin(GL_QUADS);
		glNormal3f(0, 0, 1);
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x, y, z); // 1
		glEnd();
		// face 5
		glBegin(GL_QUADS);
		glNormal3f(0, 0, -1);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x, y, z+1); // 0
		glVertex3f(x, y+1, z+1); // 3
		glEnd();
		
	}
	glColor4f(1.0, 1.0, 1.0, 1.0);
}
// helper function to rendering volume (with edge)
void kernel_draw_voxels_edge(int* posxyz, int npos, float* val, int nval, float* valthr, int nthr,  float thres){
	int ind, n, x, y, z;
	float r, g, b, l;
	for (n=0; n<nthr; ++n) {
		ind = 3 * n;
		x = posxyz[ind];
		y = posxyz[ind+1];
		z = posxyz[ind+2];
		r = val[ind];
		g = val[ind+1];
		b = val[ind+2];
		l = valthr[n];
		if (l <= thres) {continue;}
		// face 0
		glColor4f(r, g, b, l);
		glBegin(GL_QUADS);
		glNormal3f(-1, 0, 0);
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y+1.0, z); // 2
		glVertex3f(x, y+1.0, z+1.0); // 3
		glVertex3f(x, y, z+1.0); // 4
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y+1.0, z); // 2
		glVertex3f(x, y+1.0, z+1.0); // 3
		glVertex3f(x, y, z+1.0); // 4
		glEnd();
		// face 1
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(0, 1, 0);
		glVertex3f(x, y+1, z+1); // 3
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y+1, z+1); // 7
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glNormal3f(0, 1, 0);
		glVertex3f(x, y+1, z+1); // 3
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y+1, z+1); // 7
		glEnd();
		// face 2
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(1, 0, 0);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y, z+1); // 4
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glNormal3f(1, 0, 0);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y, z+1); // 4
		glEnd();
		// face 3
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(0, -1, 0);
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y, z+1); // 0
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glNormal3f(0, -1, 0);
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y, z+1); // 0
		glEnd();
		// face 4
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(0, 0, 1);
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x, y, z); // 1
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glNormal3f(0, 0, 1);
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x, y, z); // 1
		glEnd();
		// face 5
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(0, 0, -1);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x, y, z+1); // 0
		glVertex3f(x, y+1, z+1); // 3
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glNormal3f(0, 0, -1);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x, y, z+1); // 0
		glVertex3f(x, y+1, z+1); // 3
		glEnd();
	}
	glColor4f(1.0, 1.0, 1.0, 1.0);
}

// helper to rendering image with OpenGL
void kernel_draw_pixels(float* mapr, int him, int wim, float* mapg, int himg, int wimg, float* mapb, int himb, int wimb) {
	int i, j;
	int npix = him * wim;
	float val;
	int ct = 0;
	glPointSize(1.0f);
	for (i=0; i<him; ++i) {
		for (j=0; j<wim; ++j) {
			//if (val == 0.0f) {continue;}
			glBegin(GL_POINTS);
			glColor3f(mapr[ct], mapg[ct], mapb[ct]);
			glVertex2i(j, i);
			glEnd();
			++ct;
		}
	}
	glColor3f(1.0f, 1.0f, 1.0f);
}

// helper function to colormap image (used in OpenGL MIP rendering)
void kernel_color_image(float* im, int him, int wim,
						float* mapr, int him1, int wim1, float* mapg, int him2, int wim2, float* mapb, int him3, int wim3,
						float* lutr, int him4, float* lutg, int him5, float* lutb, int him6) {
	float val;
	int i, j, ind, pos;
	for (i=0; i<him; ++i) {
		for (j=0; j<wim; ++j) {
			pos = i*wim + j;
			val = im[pos];
			val *= 255.0;
			ind = (int)val;
			mapr[pos] = lutr[ind];
			mapg[pos] = lutg[ind];
			mapb[pos] = lutb[ind];
		}
	}

}


/********************************************************************************
 * GENERAL      line drawing
 ********************************************************************************/

// Draw a line in 2D space by Digital Differential Analyzer method (modified version to 1D)
void kernel_draw_2D_line_DDA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val) {
	int length, i;
	double x, y;
	double xinc, yinc;
	
	length = abs(x2 - x1);
	if (abs(y2 - y1) > length) {length = abs(y2 - y1);}
	xinc = (double)(x2 - x1) / (double) length;
	yinc = (double)(y2 - y1) / (double) length;
	x    = x1 + 0.5;
	y    = y1 + 0.5;
	for (i=0; i<=length; ++i) {
		mat[(int)y * wx + (int)x] += val;
		x = x + xinc;
		y = y + yinc;
	}
}

// Draw lines in 2D space with DDA
void kernel_draw_2D_lines_DDA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2) {
	int length, i, n;
	float flength, val;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy;

	for (i=0; i< nx1; ++i) {
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		val  = 1 / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
		for (n=0; n<=length; ++n) {
			mat[(int)y * wx + (int)x] += val;
			x = x + xinc;
			y = y + yinc;
		}
	}
}

// Draw lines in 2D space with DDA anti-aliased version 1 pix
void kernel_draw_2D_lines_DDAA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2) {
	int length, i, n;
	float flength;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy, xint, yint;

	for (i=0; i< nx1; ++i) {
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;
		// line
		for (n=1; n<length; ++n) {
			xint = int(x);
			yint = int(y);
			mat[yint*wx + xint] += (1 - fabs(x - (xint + 0.5)));
			x = x + xinc;
			y = y + yinc;
		}
	}
}

// Draw lines in 2D space with DDA anti-aliased version 2 pix 
void kernel_draw_2D_lines_DDAA2(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2) {
	int length, i, n;
	float flength;
	float x, y, lx, ly;
	float xinc, yinc;
	int x1, y1, x2, y2, diffx, diffy, xint, yint, ind;
	float val, vd, vu;

	for (i=0; i< nx1; ++i) {
		x1 = X1[i];
		x2 = X2[i];
		y1 = Y1[i];
		y2 = Y2[i];
		diffx = x2-x1;
		diffy = y2-y1;
		lx = abs(diffx);
		ly = abs(diffy);
		length = ly;
		if (lx > length) {length = lx;}
		flength = (float)length;
		xinc = diffx / flength;
		yinc = diffy / flength;
		x = x1 + 0.5;
		y = y1 + 0.5;

		// first pixel
		xint = int(x);
		yint = int(y);
		val = 1 - fabs(x - (xint + 0.5));
		mat[yint * wx + xint] += val;
		x = x + xinc;
		y = y + yinc;
		// line
		for (n=1; n<length; ++n) {
			xint = int(x);
			yint = int(y);
			ind = yint*wx + xint;
			val = 1 - fabs(x - (xint + 0.5));
			vu = (x - xint) * 0.5;
			// vd = 0.5 - vu;
			mat[ind+1] += vu;
			mat[ind] += val;
			x = x + xinc;
			y = y + yinc;
		}
		// last pixel
		xint = int(x);
		yint = int(y);
		val = 1 - fabs(x - (xint + 0.5));
		mat[yint * wx + xint] += val;
	}
}


// Draw a line in 3D space by DDA method
void kernel_draw_3D_line_DDA(float* mat, int wz, int wy, int wx, int x1, int y1, int z1, int x2, int y2, int z2, float val) {
	int length, lengthy, lengthz, i, step;
	double x, y, z, xinc, yinc, zinc;
	step = wx * wy;
	length = abs(x2 - x1);
	lengthy = abs(y2 - y1);
	lengthz = abs(z2 - z1);
	if (lengthy > length) {length = lengthy;}
	if (lengthz > length) {length = lengthz;}

	xinc = (double)(x2 - x1) / (double) length;
	yinc = (double)(y2 - y1) / (double) length;
	zinc = (double)(z2 - z1) / (double) length;
	x    = x1 + 0.5;
	y    = y1 + 0.5;
	z    = z1 + 0.5;
	for (i=0; i<=length; ++i) {
		mat[(int)z*step + (int)y*wx + (int)x] += val;
		x = x + xinc;
		y = y + yinc;
		z = z + zinc;
	}
}

// THIS MUST BE CHANGE!!!!!!!!!
// Draw a line in 2D space by DDA with my antialiaser
#define ipart_(X) ((int) X)
#define round_(X) ((int)(((double)(X)) + 0.5))
#define fpart_(X) ((double)(X) - (double)ipart_(X))
void kernel_draw_2D_line_DDAA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val) {
	int length, i;
	double x, y;
	double xinc, yinc;
	
	length = abs(x2 - x1);
	if (abs(y2 - y1) > length) {length = abs(y2 - y1);}
	xinc = (double)(x2 - x1) / (double) length;
	yinc = (double)(y2 - y1) / (double) length;
	//x    = x1 + 0.5;
	//y    = y1 + 0.5;
	x = x1;
	y = y1;
	float hval = val / 2.0;
	float vx, vy;
	int ix, iy;
	for (i=0; i<=length; ++i) {
		vx = fpart_(x);
		vy = fpart_(y);
		ix = round_(x);
		iy = round_(y);
		mat[iy * wx + ix] += hval;
		
		if (vx > 0.5) {mat[iy * wx + ix + 1] += hval;}
		else if (vx < 0.5) {mat[iy * wx + ix - 1] += hval;}
		if (vy > 0.5) {mat[(iy + 1) * wx + ix] += hval;}
		else if (vy > 0.5) {mat[(iy - 1) * wx + ix] += hval;}
		
		x = x + xinc;
		y = y + yinc;
	}
}
#undef ipart_
#undef fpart_
#undef round_

// Draw a line in 2D space by Bresenham's Line Algorithm (modified version 1D)
void kernel_draw_2D_line_BLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val) {
	int x, y;
	int dx, dy;
	int xinc, yinc;
	int balance;
	if (x2 >= x1) {
		dx = x2 - x1;
		xinc = 1;
	} else {
		dx = x1 - x2;
		xinc = -1;
	}
	if (y2 >= y1) {
		dy = y2 - y1;
		yinc = 1;
	} else {
		dy = y1 - y2;
		yinc = -1;
	}
	x = x1;
	y = y1;
	if (dx >= dy) {
		dy <<= 1;
		balance = dy - dx;
		dx <<= 1;
		while (x != x2) {
			mat[y * wx + x] += val;
			if (balance >= 0) {
				y = y + yinc;
				balance = balance - dx;
			}
			balance = balance + dy;
			x = x + xinc;
		}
		mat[y * wx + x] += val;
	} else {
		dx <<= 1;
		balance = dx - dy;
		dy <<= 1;
		while (y != y2) {
			mat[y * wx + x] += val;
			if (balance >= 0) {
				x = x + xinc;
				balance = balance - dy;
			}
			balance = balance + dx;
			y = y + yinc;
		}
		mat[y * wx + x] += val;
	}
}

// Draw lines in 2D space by Bresenham's Line Algorithm (modified version 1D)
void kernel_draw_2D_lines_BLA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2) {
	int x, y, n;
	int x1, y1, x2, y2;
	int dx, dy;
	int xinc, yinc;
	int balance;
	float val;

	for (n=0; n<nx1; ++n) {
		x1 = X1[n];
		y1 = Y1[n];
		x2 = X2[n];
		y2 = Y2[n];

		if (x2 >= x1) {
			dx = x2 - x1;
			xinc = 1;
		} else {
			dx = x1 - x2;
			xinc = -1;
		}
		if (y2 >= y1) {
			dy = y2 - y1;
			yinc = 1;
		} else {
			dy = y1 - y2;
			yinc = -1;
		}
		
		x = x1;
		y = y1;
		if (dx >= dy) {
			val = 1 / (float)dx;
			dy <<= 1;
			balance = dy - dx;
			dx <<= 1;
			while (x != x2) {
				mat[y * wx + x] += val;
				if (balance >= 0) {
					y = y + yinc;
					balance = balance - dx;
				}
				balance = balance + dy;
				x = x + xinc;
			}
			mat[y * wx + x] += val;
		} else {
			val = 1 / (float)dy;
			dx <<= 1;
			balance = dx - dy;
			dy <<= 1;
			while (y != y2) {
				mat[y * wx + x] += val;
				if (balance >= 0) {
					x = x + xinc;
					balance = balance - dy;
				}
				balance = balance + dx;
				y = y + yinc;
			}
			mat[y * wx + x] += val;
		}
	}
}

// Draw lines in 2D space by Siddon's Line Algorithm (modified version 1D)
void kernel_draw_2D_lines_SIDDON(float* mat, int wy, int wx, float* X1, int nx1, float* Y1, int ny1, float* X2, int nx2, float* Y2, int ny2, int res, int b, int matsize) {
	int n;
	float tx, ty, px, qx, py, qy;
	int ei, ej, u, v, i, j;
	int stepi, stepj;
	float divx, divy, runx, runy, oldv, newv, val, valmax;
	float axstart, aystart, astart, pq, stepx, stepy, startl;
	for (n=0; n<nx1; ++n) {
		px = X2[n];
		py = Y2[n];
		qx = X1[n];
		qy = Y1[n];
		tx = (px-qx) * 0.4 + qx; // not 0.5 to avoid an image artefact
		ty = (py-qy) * 0.4 + qy;
		ei = int((tx-b) / (float)res);
		ej = int((ty-b) / (float)res);
		
		if (qx-tx>0) {
			u=ei+1;
			stepi=1;
		}
		if (qx-tx<0) {
			u=ei;
			stepi=-1;
		}
		if (qx-tx==0) {
			u=ei;
			stepi=0;
		}
		if (qy-ty>0) {
			v=ej+1;
			stepj=1;
		}
		if (qy-ty<0) {
			v=ej;
			stepj=-1;
		}
		if (qy-ty==0) {
			v=ej;
			stepj=0;
		}
		if (qx==px) {divx=1.0;}
		else {divx = float(qx-px);}
		if (qy==py) {divy=1.0;}
		else {divy = float(qy-py);}

		axstart = ((u*res)+b-px) / divx;
		aystart = ((v*res)+b-py) / divy;
		astart = aystart;
		if (axstart > aystart) {astart = axstart;}
		pq = sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py));
		stepx = fabs((res*pq / divx));
		stepy = fabs((res*pq / divy));
		startl = astart * pq;
		valmax = stepx;
		if (stepy < valmax) {valmax = stepy;}
		valmax = valmax + valmax*0.01f;
		//valmax = sqrt(stepx * stepx + stepy * stepy);

		// first half-ray
		runx = axstart * pq;
		runy = aystart * pq;
		i = ei;
		j = ej;
		if (runx == startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy == startl) {
			j += stepj;
			runy += stepy;
		}
		oldv = startl;
		while (i>=0 && j>=0 && i<matsize && j<matsize) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			mat[j * wx + i] += val;
			oldv = newv;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
		}

		// second half-ray
		if (px-tx>0) {stepi=1;}
		if (px-tx<0) {stepi=-1;}
		if (py-ty>0) {stepj=1;}
		if (py-ty<0) {stepj=-1;}
		runx = axstart * pq;
		runy = aystart * pq;
		i = ei;
		j = ej;
		if (runx==startl) {
			i += stepi;
			runx += stepx;
		}
		if (runy==startl) {
			j += stepj;
			runy += stepy;
		}
		oldv = startl;
		mat[ej * wx + ei] += valmax;
		while (i>=0 && j>=0 && i<matsize && j<matsize) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			val = fabs(newv - oldv);
			if (val > valmax) {val = valmax;}
			mat[j * wx + i] += val;
			oldv = newv;
			if (runx == newv) {
				i += stepi;
				runx += stepx;
			}
			if (runy == newv) {
				j += stepj;
				runy += stepy;
			}
		}
	}
}


// Draw a line in 2D space by Wu's Antialiasing Line Algorithm (modified version 1D)
#define ipart_(X) ((int) X)
#define round_(X) ((int)(((double)(X)) + 0.5))
#define fpart_(X) ((double)(X) - (double)ipart_(X))
#define rfpart_(X) (1.0 - fpart_(X))
#define swap_(a, b) do{ __typeof__(a) tmp; tmp = a; a = b; b = tmp; }while(0)
void kernel_draw_2D_line_WALA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val) {
	double dx = (double)x2 - (double)x1;
	double dy = (double)y2 - (double)y1;

	if (fabs(dx) > fabs(dy)) {
		if (x2 < x1) {
			swap_(x1, x2);
			swap_(y1, y2);
		}
		
	    double gradient = dy / dx;
		double xend = round_(x1);
		double yend = y1 + gradient * (xend - x1);
		double xgap = rfpart_(x1 + 0.5);
		int xpxl1 = xend;
		int ypxl1 = ipart_(yend);
		mat[ypxl1 * wx + xpxl1] += (rfpart_(yend) * xgap * val);
		mat[(ypxl1 + 1) * wx + xpxl1] += (fpart_(yend) * xgap * val);
		double intery = yend + gradient;
		
		xend = round_(x2);
		yend = y2 + gradient*(xend - x2);
		xgap = fpart_(x2+0.5);
		int xpxl2 = xend;
		int ypxl2 = ipart_(yend);
		mat[ypxl2 * wx + xpxl2] += (rfpart_(yend) * xgap * val);
		mat[(ypxl2 + 1) * wx + xpxl2] += (fpart_(yend) * xgap * val);
		int x;
		for (x=xpxl1+1; x <= (xpxl2-1); x++) {
			mat[ipart_(intery) * wx + x] += (rfpart_(intery) * val);
			mat[(ipart_(intery) + 1) * wx + x] += (fpart_(intery) * val);
			intery += gradient;
		}
	} else {
		if (y2 < y1) {
			swap_(x1, x2);
			swap_(y1, y2);
		}
		double gradient = dx / dy;
		double yend = round_(y1);
		double xend = x1 + gradient*(yend - y1);
		double ygap = rfpart_(y1 + 0.5);
		int ypxl1 = yend;
		int xpxl1 = ipart_(xend);
		mat[ypxl1 * wx + xpxl1] += (rfpart_(xend) * ygap * val);
		mat[(ypxl1 + 1) * wx + xpxl1] += (fpart_(xend) * ygap * val);
		double interx = xend + gradient;

		yend = round_(y2);
		xend = x2 + gradient*(yend - y2);
		ygap = fpart_(y2+0.5);
		int ypxl2 = yend;
		int xpxl2 = ipart_(xend);
		mat[ypxl2 * wx + xpxl2] += (rfpart_(xend) * ygap * val);
		mat[(ypxl2 + 1) * wx + xpxl2] += (fpart_(xend) * ygap * val);

		int y;
		for(y=ypxl1+1; y <= (ypxl2-1); y++) {
			mat[y * wx + ipart_(interx)] += (rfpart_(interx) * val);
			mat[y * wx + ipart_(interx) + 1] += (fpart_(interx) * val);
			interx += gradient;
		}
	}
}
#undef swap_
#undef ipart_
#undef fpart_
#undef round_
#undef rfpart_

// Draw a line in 2D space by Wu's Line Algorithm (modified version 1D)
void kernel_draw_2D_line_WLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val) {
	int dy = y2 - y1;
	int dx = x2 - x1;
	int stepx, stepy;

	if (dy < 0) { dy = -dy;  stepy = -1; } else { stepy = 1; }
	if (dx < 0) { dx = -dx;  stepx = -1; } else { stepx = 1; }

	mat[y1 * wx + x1] += val;
	mat[y2 * wx + x2] += val;
	if (dx > dy) {
		int length = (dx - 1) >> 2;
		int extras = (dx - 1) & 3;
		int incr2 = (dy << 2) - (dx << 1);
		if (incr2 < 0) {
			int c = dy << 1;
			int incr1 = c << 1;
			int d =  incr1 - dx;
			for (int i = 0; i < length; i++) {
				x1 = x1 + stepx;
				x2 = x2 - stepx;
				if (d < 0) {                    // Pattern:
					mat[y1 * wx + x1] += val;   //
					x1 = x1 + stepx;            // x o o
					mat[y1 * wx + x1] += val;
					mat[y2 * wx + x2] += val;
					x2 = x2 - stepx;
					mat[y2 * wx + x2] += val;
					d += incr1;
				} else {
					if (d < c) {                                 // Pattern:
						mat[y1 * wx + x1] += val;                //       o
						x1 = x1 + stepx;                         //   x o
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						mat[y2 * wx + x2] += val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
						
					} else {
						y1 = y1 + stepy;                      // Pattern
						mat[y1 * wx + x1] += val;             //    o o
						x1 = x1 + stepx;                      //  x
						mat[y1 * wx + x1] += val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
					}
					d += incr2;
				}
			}
			if (extras > 0) {
				if (d < 0) {
					x1 = x1 + stepx;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
					}
				} else 
                if (d < c) {
					x1 = x1 + stepx;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
					}
				} else {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
                }
			}
		} else {
		    int c = (dy - dx) << 1;
			int incr1 = c << 1;
			int d =  incr1 + dx;
			for (int i = 0; i < length; i++) {
				x1 = x1 + stepx;
				x2 = x2 - stepx;
				if (d > 0) {
					y1 = y1 + stepy;           // Pattern
					mat[y1 * wx + x1] += val;  //      o
					x1 = x1 + stepx;           //    o
					y1 = y1 + stepy;           //   x
					mat[y1 * wx + x1] += val;
					y2 = y2 - stepy;
					mat[y2 * wx + x2] += val;
					x2 = x2 - stepx;
					y2 = y2 - stepy;
					mat[y2 * wx + x2] += val;
					d += incr1;
				} else {
					if (d < c) {
						mat[y1 * wx + x1] += val;  // Pattern
						x1 = x1 + stepx;           //      o
						y1 = y1 + stepy;           //  x o
						mat[y1 * wx + x1] += val;
						mat[y2 * wx + x2] += val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					} else {
						y1 = y1 + stepy;          // Pattern
						mat[y1 * wx + x1] += val; //    o  o
						x1 = x1 + stepx;          //  x
						mat[y1 * wx + x1] += val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
					}
					d += incr2;
				}
			}
			if (extras > 0) {
				if (d > 0) {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
				} else 
                if (d < c) {
					x1 = x1 + stepx;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
					}
				} else {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						if (d > c) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						} else {
							x2 = x2 - stepx;
							mat[y2 * wx + x2] += val;
						}
					}
				}
			}
		}
    } else {
	    int length = (dy - 1) >> 2;
		int extras = (dy - 1) & 3;
		int incr2 = (dx << 2) - (dy << 1);
		if (incr2 < 0) {
			int c = dx << 1;
			int incr1 = c << 1;
			int d =  incr1 - dy;
			for (int i = 0; i < length; i++) {
				y1 = y1 + stepy;
				y2 = y2 - stepy;
				if (d < 0) {
					mat[y1 * wx + x1] += val;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					mat[y2 * wx + x2] += val;
					y2 = y2 - stepy;
					mat[y2 * wx + x2] += val;
					d += incr1;
				} else {
					if (d < c) {
						mat[y1 * wx + x1] += val;
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						mat[y2 * wx + x2] += val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					} else {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
					d += incr2;
				}
			}
			if (extras > 0) {
				if (d < 0) {
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
				} else 
                if (d < c) {
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
                } else {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
                }
			}
        } else {
			int c = (dx - dy) << 1;
			int incr1 = c << 1;
			int d =  incr1 + dy;
			for (int i = 0; i < length; i++) {
				y1 = y1 + stepy;
				y2 = y2 - stepy;
				if (d > 0) {
					x1 = x1 + stepx;
					mat[y1 * wx + x1] += val;
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					x2 = x2 - stepx;
					mat[y2 * wx + x2] += val;
					x2 = x2 - stepx;
					y2 = y2 - stepy;
					mat[y2 * wx + x2] += val;
					d += incr1;
				} else {
					if (d < c) {
						mat[y1 * wx + x1] += val;
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						mat[y2 * wx + x2] += val; 
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					} else {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
					d += incr2;
				}
			}
			if (extras > 0) {
				if (d > 0) {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
				} else
                if (d < c) {
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
                    if (extras > 2) {
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
				} else {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						if (d > c)  {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						} else {
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						}
					}
				}
			}
		}
	}
}

// Draw a list of lines in 2D space by Wu's Line Algorithm (modified version 1D)
void kernel_draw_2D_lines_WLA(float* mat, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2) {
	int dx, dy, stepx, stepy, n;
	int length, extras, incr2, incr1, c, d, i;
	int x1, y1, x2, y2;
	float val;
	for (n=0; n<nx1; ++n) {
		x1 = X1[n];
		y1 = Y1[n];
		x2 = X2[n];
		y2 = Y2[n];
	    dy = y2 - y1;
		dx = x2 - x1;
	
		if (dy < 0) { dy = -dy;  stepy = -1; } else { stepy = 1; }
		if (dx < 0) { dx = -dx;  stepx = -1; } else { stepx = 1; }
		if (dx > dy) {val = 1 / float(dx);}
		else {val = 1 / float(dy);}
	
		mat[y1 * wx + x1] += val;
		mat[y2 * wx + x2] += val;
		if (dx > dy) {
			length = (dx - 1) >> 2;
			extras = (dx - 1) & 3;
			incr2 = (dy << 2) - (dx << 1);
			if (incr2 < 0) {
				c = dy << 1;
				incr1 = c << 1;
				d =  incr1 - dx;
				for (i = 0; i < length; i++) {
					x1 = x1 + stepx;
					x2 = x2 - stepx;
					if (d < 0) {                    // Pattern:
						mat[y1 * wx + x1] += val;   //
						x1 = x1 + stepx;            // x o o
						mat[y1 * wx + x1] += val;
						mat[y2 * wx + x2] += val;
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
						d += incr1;
					} else {
						if (d < c) {                                 // Pattern:
							mat[y1 * wx + x1] += val;                //       o
							x1 = x1 + stepx;                         //   x o
							y1 = y1 + stepy;
							mat[y1 * wx + x1] += val;
							mat[y2 * wx + x2] += val;
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
							
						} else {
							y1 = y1 + stepy;                      // Pattern
							mat[y1 * wx + x1] += val;             //    o o
							x1 = x1 + stepx;                      //  x
							mat[y1 * wx + x1] += val;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
							x2 = x2 - stepx;
							mat[y2 * wx + x2] += val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d < 0) {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
						if (extras > 1) {
							x1 = x1 + stepx;
							mat[y1 * wx + x1] += val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							mat[y2 * wx + x2] += val;
						}
					} else 
	                if (d < c) {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] += val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							mat[y2 * wx + x2] += val;
						}
					} else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						if (extras > 1) {
							x1 = x1 + stepx;
							mat[y1 * wx + x1] += val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						}
	                }
				}
			} else {
			    c = (dy - dx) << 1;
				incr1 = c << 1;
				d =  incr1 + dx;
				for (i = 0; i < length; i++) {
					x1 = x1 + stepx;
					x2 = x2 - stepx;
					if (d > 0) {
						y1 = y1 + stepy;           // Pattern
						mat[y1 * wx + x1] += val;  //      o
						x1 = x1 + stepx;           //    o
						y1 = y1 + stepy;           //   x
						mat[y1 * wx + x1] += val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
						d += incr1;
					} else {
						if (d < c) {
							mat[y1 * wx + x1] += val;  // Pattern
							x1 = x1 + stepx;           //      o
							y1 = y1 + stepy;           //  x o
							mat[y1 * wx + x1] += val;
							mat[y2 * wx + x2] += val;
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						} else {
							y1 = y1 + stepy;          // Pattern
							mat[y1 * wx + x1] += val; //    o  o
							x1 = x1 + stepx;          //  x
							mat[y1 * wx + x1] += val;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
							x2 = x2 - stepx;
							mat[y2 * wx + x2] += val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d > 0) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] += val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						}
					} else 
	                if (d < c) {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] += val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							mat[y2 * wx + x2] += val;
						}
					} else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						if (extras > 1) {
							x1 = x1 + stepx;
							mat[y1 * wx + x1] += val;
						}
						if (extras > 2) {
							if (d > c) {
								x2 = x2 - stepx;
								y2 = y2 - stepy;
								mat[y2 * wx + x2] += val;
							} else {
								x2 = x2 - stepx;
								mat[y2 * wx + x2] += val;
							}
						}
					}
				}
			}
	    } else {
		    length = (dy - 1) >> 2;
			extras = (dy - 1) & 3;
			incr2 = (dx << 2) - (dy << 1);
			if (incr2 < 0) {
				c = dx << 1;
				incr1 = c << 1;
				d =  incr1 - dy;
				for (i = 0; i < length; i++) {
					y1 = y1 + stepy;
					y2 = y2 - stepy;
					if (d < 0) {
						mat[y1 * wx + x1] += val;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						mat[y2 * wx + x2] += val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
						d += incr1;
					} else {
						if (d < c) {
							mat[y1 * wx + x1] += val;
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] += val;
							mat[y2 * wx + x2] += val;
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						} else {
							x1 = x1 + stepx;
							mat[y1 * wx + x1] += val;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] += val;
							x2 = x2 - stepx;
							mat[y2 * wx + x2] += val;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d < 0) {
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						if (extras > 1) {
							y1 = y1 + stepy;
							mat[y1 * wx + x1] += val;
						}
						if (extras > 2) {
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						}
					} else 
	                if (d < c) {
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] += val;
						}
						if (extras > 2) {
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						}
	                } else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						if (extras > 1) {
							y1 = y1 + stepy;
							mat[y1 * wx + x1] += val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						}
	                }
				}
	        } else {
				c = (dx - dy) << 1;
				incr1 = c << 1;
				d =  incr1 + dy;
				for (i = 0; i < length; i++) {
					y1 = y1 + stepy;
					y2 = y2 - stepy;
					if (d > 0) {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
						d += incr1;
					} else {
						if (d < c) {
							mat[y1 * wx + x1] += val;
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] += val;
							mat[y2 * wx + x2] += val; 
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						} else {
							x1 = x1 + stepx;
							mat[y1 * wx + x1] += val;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] += val;
							x2 = x2 - stepx;
							mat[y2 * wx + x2] += val;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						}
						d += incr2;
					}
				}
				if (extras > 0) {
					if (d > 0) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] += val;
						}
						if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						}
					} else
	                if (d < c) {
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						if (extras > 1) {
							x1 = x1 + stepx;
							y1 = y1 + stepy;
							mat[y1 * wx + x1] += val;
						}
	                    if (extras > 2) {
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						}
					} else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						if (extras > 1) {
							y1 = y1 + stepy;
							mat[y1 * wx + x1] += val;
						}
						if (extras > 2) {
							if (d > c)  {
								x2 = x2 - stepx;
								y2 = y2 - stepy;
								mat[y2 * wx + x2] += val;
							} else {
								y2 = y2 - stepy;
								mat[y2 * wx + x2] += val;
							}
						}
					}
				}
			}
		}
	}
}



/**************************************************************
 * 2D PET SCAN      resconstruction
 **************************************************************/

// EM-ML algorithm, only one iteration (bin mode)
void kernel_pet2D_EMML_iter(float* SRM, int nlor, int npix, float* S, int nbs, float* im, int npixim, int* LOR_val, int nlorval) {
	int i, j, ind;
	float qi, buf, f;
	float* Q = (float*)malloc(nlor * sizeof(float));

	// compute expected value
	for (i=0; i<nlor; ++i) {
		qi = 0.0;
		ind = i * npix;
		for (j=0; j<npix; ++j) {qi += (SRM[ind+j] * im[j]);}
		Q[i] = qi;
	}

	// update pixel
	for (j=0; j<npix; ++j) {
		buf = im[j];
		
		if (buf != 0) {
			f = 0.0;
			for (i=0; i<nlor; ++i) {
				f += (LOR_val[i] * SRM[i * npix + j] / Q[i]);
			}
			im[j] = buf / S[j] * f;
		}
	}
	free(Q);
}

// EM-ML algorithm, only one iteration (list-mode), Naive implementation as define by the method
void kernel_pet2D_LM_EMML_iter(float* SRM, int nlor, int npix, float* S, int nbs, float* im, int npixim) {
	int i, j, ind;
	float qi, buf, f;
	float* Q = (float*)malloc(nlor * sizeof(float));

	// compute expected value
	for (i=0; i<nlor; ++i) {
		qi = 0.0;
		ind = i * npix;
		for (j=0; j<npix; ++j) {qi += (SRM[ind+j] * im[j]);}
		if (qi == 0.0) {qi = 1.0f;}
		Q[i] = qi;
	}

	// update pixel
	for (j=0; j<npix; ++j) {
		buf = im[j];
		if (buf != 0) {
			f = 0.0;
			for (i=0; i<nlor; ++i) {
				f += (SRM[i * npix + j] / Q[i]);
			}
			im[j] = buf / S[j] * f;
		}
	}
	free(Q);
}

// EM-ML algorithm with sparse matrix (COO), only one iteration (list-mode), matrix operation
void kernel_pet2D_LM_EMML_COO_iter_mat(float* SRMvals, int nvals, int* SRMrows, int nrows, int* SRMcols, int ncols, float* S, int nbs, float* im, int npix, int nevents) {
	int i, j, ind;
	float buf;
	float* Q = (float*)malloc(nevents * sizeof(float));
	float* F = (float*)malloc(npix * sizeof(float));

	// init Q and F
	for (i=0; i<nevents; ++i) {Q[i] = 0.0f;}
	for (i=0; i<npix; ++i) {F[i] = 0.0f;}
	
	// Sparse matrix multiplication Q = SRM * im
	for (i=0; i<nvals; ++i) {
		Q[SRMrows[i]] += (SRMvals[i] * im[SRMcols[i]]);
	}
	// Sparse matrix operation F = SRM^T / Q
	for (i=0; i<nvals; ++i) {
		F[SRMcols[i]] += (SRMvals[i] / Q[SRMrows[i]]);
	}
	// update pixel
	for (j=0; j<npix; ++j) {
		buf = im[j];
		if (buf != 0) {
			im[j] = buf / S[j] * F[j];
		}
	}
	free(F);
	free(Q);
}

// EM-ML algorithm with sparse matrix (COO), only one iteration (list-mode), naive method scalar operation
void kernel_pet2D_LM_EMML_COO_iter_vec(float* SRMvals, int nvals, int* SRMrows, int nrows, int* SRMcols, int ncols, float* S, int nbs, float* im, int npix, int nevents) {
	int i, j, ind;
	float buf, f;
	float* Q = (float*)malloc(nevents * sizeof(float));

	// init Q and F
	for (i=0; i<nevents; ++i) {Q[i] = 0.0f;}
	
	// Sparse matrix multiplication Q = SRM * im
	for (i=0; i<nvals; ++i) {
		Q[SRMrows[i]] += (SRMvals[i] * im[SRMcols[i]]);
	}
	// update pixel
	for (j=0; j<npix; ++j) {
		printf("%i\n", j);
		buf = im[j];
		if (buf != 0) {
			f = 0.0;
			for (i=0; i<ncols; ++i) {
				if (SRMcols[i] == j) {
					f += (SRMvals[i] / Q[SRMrows[i]]);
				}
			}
			im[j] = buf / S[j] * f;
		}
	}
	free(Q);
}


// EM-ML algorithm with sparse matrix (ELL)
void kernel_pet2D_LM_EMML_ELL_iter(float* SRMvals, int nivals, int njvals, int* SRMcols, int nicols, int njcols, float* S, int ns, float* im, int npix) {
	int i, j, ind, vcol;
	float buf, sum;
	float* Q = (float*)calloc(nivals, sizeof(float));
	float* F = (float*)calloc(npix, sizeof(float));

	// Sparse matrix operation Q = SRM * im
	for (i=0; i<nivals; ++i) {
		ind = i * njvals;
		vcol = SRMcols[ind];
		j = 0;
		sum = 0.0f;
		while (vcol != -1) {
			sum += (SRMvals[ind+j] * im[vcol]);
			++j;
			vcol = SRMcols[ind+j];
		}
		Q[i] = sum;
	}
	// Sparse matrix operation F = SRM^T / Q
	for (i=0; i<nivals; ++i) {
		ind = i * njvals;
		vcol = SRMcols[ind];
		j = 0;
		while (vcol != -1) {
			F[vcol] += (SRMvals[ind+j] / Q[i]);
			++j;
			vcol = SRMcols[ind+j];
		}
	}
	// update pixel
	for (j=0; j<npix; ++j) {
		buf = im[j];
		if (buf != 0) {
			im[j] = buf / S[j] * F[j];
		}
	}
	free(Q);
	free(F);
}

// EM-ML algorithm, only one iteration MPI version
void kernel_pet2D_EMML_iter_MPI(float* SRM, int nlor, int npix, float* S, int nbs, float* im, int npixim, int* LOR_val, int nlorval, int N_start, int N_stop) {
	int i, j, ind;
	float qi, buf, f;
	float* Q = (float*)malloc(nlor * sizeof(float));
	
	// compute expected value
	for (i=0; i<nlor; ++i) {
		qi = 0.0;
		ind = i * npix;
		for (j=0; j<npix; ++j) {qi += (SRM[ind+j] * im[j]);}
		Q[i] = qi;
	}

	// update pixel
	for (j=N_start; j<N_stop; ++j) {
		buf = im[j];
		
		if (buf != 0) {
			f = 0.0;
			for (i=0; i<nlor; ++i) {
				f += (LOR_val[i] * SRM[i * npix + j] / Q[i]);
			}
			im[j] = buf / S[j] * f;
		}
	}
	free(Q);
}

// EM-ML algorithm, all iterations GPU version
void kernel_pet2D_EMML_cuda(float* SRM, int nlor, int npix, float* im, int npixim, int* LOR_val, int nval, float* S, int ns, int maxit) {
	kernel_pet2D_EMML_wrap_cuda(SRM, nlor, npix, im, npixim, LOR_val, nval, S, ns, maxit);
}


// List mode 2D rexonstruction with DDA and ELL format, all iterations are perform on GPU
void kernel_pet2D_LM_EMML_DDA_ELL_cuda(int* x1, int nx1, int* y1, int ny1, int* x2, int nx2, int* y2, int ny2, float* im, int nim, float* S, int ns, int wsrm, int wim, int maxite) {
	kernel_pet2D_LM_EMML_DDA_ELL_wrap_cuda(x1, nx1, y1, ny1, x2, nx2, y2, ny2, im, nim, S, ns, wsrm, wim, maxite);
}

// Compute first image in 2D-LM-OSEM reconstruction (from IM, x, y build SRM in ELL format then compute IM+=IM)
void kernel_pet2D_IM_SRM_DDA_ELL_cuda(int* x1, int nx1, int* y1, int ny1, int* x2, int nx2, int* y2, int ny2, float* im, int nim, int wsrm, int wim) {
	kernel_pet2D_IM_SRM_DDA_ELL_wrap_cuda(x1, nx1, y1, ny1, x2, nx2, y2, ny2, im, nim, wsrm, wim);
}

// Update image for the 2D-LM-OSEM reconstruction (from x, y, IM and S, build SRM in ELL format then update IM)
void kernel_pet2D_IM_SRM_DDA_ELL_iter_cuda(int* x1, int nx1, int* y1, int ny1, int* x2, int nx2, int* y2, int ny2, float* S, int ns, float* im, int nim, int wsrm, int wim) {
	kernel_pet2D_IM_SRM_DDA_ELL_iter_wrap_cuda(x1, nx1, y1, ny1, x2, nx2, y2, ny2, S, ns, im, nim, wsrm, wim);
}

// Compute first image in 3D-LM-OSEM reconstruction (from IM, x, y build SRM in ELL format then compute IM+=IM)
void kernel_pet3D_IM_SRM_DDA_ELL_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1, unsigned short int* z1, int nz1,
									  unsigned short int* x2, int nx2, unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
									  float* im, int nim, int wsrm, int wim, int ID) {
	kernel_pet3D_IM_SRM_DDA_ELL_wrap_cuda(x1, nx1, y1, ny1, z1, nz1, x2, nx2, y2, ny2, z2, nz2, im, nim, wsrm, wim, ID);
}

// Update image for the 3D-LM-OSEM reconstruction (from x, y, IM and S, build SRM in ELL format then return F)
void kernel_pet3D_IM_SRM_DDA_ELL_iter_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1, unsigned short int* z1, int nz1,
										   unsigned short int* x2, int nx2, unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
										   float* im, int nim, float* F, int nf, int wsrm, int wim, int ID) {
	kernel_pet3D_IM_SRM_DDA_ELL_iter_wrap_cuda(x1, nx1, y1, ny1, z1, nz1, x2, nx2, y2, ny2, z2, nz2, im, nim, F, nf, wsrm, wim, ID);
}


/**************************************************************
 * Utils
 **************************************************************/

// Convert dense matrix to sparse one with COO format
void kernel_matrix_mat2coo(float* mat, int ni, int nj, float* vals, int nvals, int* rows, int nrows, int* cols, int ncols, int roffset, int coffset) {
	// roffset and coffset are rows and colums shiftment, if mat is a tile of a big matrix indexes must adjust
	int i, j, ind;
	int ct = 0;
	float buf;
	for (i=0; i<ni; ++i) {
		ind = i*nj;
		for (j=0; j<nj; ++j) {
			buf = mat[ind + j];
			if (buf != 0.0f) {
				rows[ct] = i + roffset;
				cols[ct] = j + coffset;
				vals[ct] = buf;
				++ct;
			}
		}
		
	}
}

// Compute col sum of COO matrix
void kernel_matrix_coo_sumcol(float* vals, int nvals, int* cols, int ncols, float* im, int npix) {
	int n;
	for (n=0; n<nvals; ++n) {
		im[cols[n]] += vals[n];
	}
}

// Compute spmv matrix/vector multiplication with sparse COO matrix
void kernel_matrix_coo_spmv(float* vals, int nvals, int* cols, int ncols, int* rows, int nrows, float* y, int ny, float* res, int nres) {
	int n;
	for (n=0; n<nvals; ++n) {
		res[rows[n]] += (vals[n] * y[cols[n]]);
	}
}

// Compute spmtv (t for transpose) matrix/vector multiplication with sparse COO matrix 
void kernel_matrix_coo_spmtv(float* vals, int nvals, int* cols, int ncols, int* rows, int nrows, float* y, int ny, float* res, int nres) {
	int n;
	for (n=0; n<nvals; ++n) {
		res[cols[n]] += (vals[n] * y[rows[n]]);
	}
}

// Convert dense matrix to sparse one with CSR format
void kernel_matrix_mat2csr(float* mat, int ni, int nj, float* vals, int nvals, int* ptr, int nptr, int* cols, int ncols) {
	int i, j, ind;
	int ct = 0;
	float buf;
	for (i=0; i<ni; ++i) {
		ptr[i] = -1;
		ind = i*nj;
		for (j=0; j<nj; ++j) {
			buf = mat[ind + j];
			if (buf != 0) {
				if (ptr[i] == -1) {ptr[i] = ct;}
				cols[ct] = j;
				vals[ct] = buf;
				++ct;
			}
		}
		if (ptr[i] == -1) {ptr[i] = ct;}
	}
	ptr[ni] = nvals;
}

// Compute col sum of CSR matrix
void kernel_matrix_csr_sumcol(float* vals, int nvals, int* cols, int ncols, float* im, int npix) {
	int n;
	for (n=0; n<nvals; ++n) {
		im[cols[n]] += vals[n];
	}
}


// Compute spmv matrix/vector multiplication with sparse CSR matrix
void kernel_matrix_csr_spmv(float* vals, int nvals, int* cols, int ncols, int* ptrs, int nptrs, float* y, int ny, float* res, int nres) {
	int iptr, k;
	for (iptr=0; iptr<(nptrs-1); ++iptr) {
		for (k=ptrs[iptr]; k<ptrs[iptr+1]; ++k) {
			res[iptr] += (vals[k] * y[cols[k]]);
		}
	}
}

// Compute spmtv (t for transpose) matrix/vector multiplication with sparse CSR matrix 
void kernel_matrix_csr_spmtv(float* vals, int nvals, int* cols, int ncols, int* ptrs, int nptrs, float* y, int ny, float* res, int nres) {
	int iptr, k;
	for (iptr=0; iptr<(nptrs-1); ++iptr) {
		for (k=ptrs[iptr]; k<ptrs[iptr+1]; ++k) {
			res[cols[k]] += (vals[k] * y[iptr]);
		}
	}
}

// Convert dense matrix to sparse one with ELL format
void kernel_matrix_mat2ell(float* mat, int ni, int nj, float* vals, int nivals, int njvals, int* cols, int nicols, int njcols) {
	int i, j, ind1, ind2, icol;
	float buf;
	for (i=0; i<ni; ++i) {
		ind1 = i*nj;
		ind2 = i*njvals;
		icol = 0;
		for (j=0; j<nj; ++j) {
			buf = mat[ind1+j];
			if (buf != 0.0f) {
				vals[ind2+icol] = buf;
				cols[ind2+icol] = j;
				++icol;
			}
		}
		cols[ind2+icol] = -1; // eof
	}
}

// Compute col sum of ELL matrix
void kernel_matrix_ell_sumcol(float* vals, int niv, int njv, int* cols, int nic, int njc, float* im, int npix) {
	int i, j, ind, vcol;
	for (i=0; i<niv; ++i) {
		ind = i * njv;
		vcol = cols[ind];
		j = 0;
		while (vcol != -1) {
			im[vcol] += vals[ind+j];
			++j;
			vcol = cols[ind+j];
		}
	}
}

// Compute spmv matrix/vector multiplication with sparse ELL matrix
void kernel_matrix_ell_spmv(float* vals, int niv, int njv, int* cols, int nic, int njc, float* y, int ny, float* res, int nres) {
	int i, j, ind, vcol;
	float sum;
	for (i=0; i<niv; ++i) {
		ind = i * njv;
		vcol = cols[ind];
		j = 0;
		sum = 0.0f;
		while (vcol != -1) {
			sum += (vals[ind+j] * y[vcol]);
			++j;
			vcol = cols[ind+j];
		}
		res[i] = sum;
	}
}

// Compute spmv matrix/vector multiplication with sparse ELL matrix using GPU
void kernel_matrix_ell_spmv_cuda(float* vals, int niv, int njv, int* cols, int nic, int njc, float* y, int ny, float* res, int nres) {
	kernel_matrix_ell_spmv_wrap_cuda(vals, niv, njv, cols, nic, njc, y, ny, res, nres);
}

// Compute spmtv matrix/vector multiplication with sparse ELL matrix
void kernel_matrix_ell_spmtv(float* vals, int niv, int njv, int* cols, int nic, int njc, float* y, int ny, float* res, int nres) {
	int i, j, ind, vcol;
	for (i=0; i<niv; ++i) {
		ind = i * njv;
		vcol = cols[ind];
		j = 0;
		while (vcol != -1) {
			res[vcol] += (vals[ind+j] * y[i]);
			++j;
			vcol = cols[ind+j];
		}
	}
}

// Compute spmv matrix/vector multiplication
void kernel_matrix_spmv(float* mat, int ni, int nj, float* y, int ny, float* res, int nres) {
	int i, j, ind;
	float sum;
	for (i=0; i<ni; ++i) {
		sum = 0.0f;
		ind = i*nj;
		for (j=0; j<nj; ++j) {
			sum += (mat[ind+j] * y[j]);
		}
		res[i] = sum;
	}
}

// Compute spmtv matrix/vector multiplication
void kernel_matrix_spmtv(float* mat, int ni, int nj, float* y, int ny, float* res, int nres) {
	int i, j, ind;
	float sum;
	for (j=0; j<nj; ++j) {
		sum = 0.0f;
		for (i=0; i<ni; ++i) {
			sum += (mat[i*nj + j] * y[i]);
		}
		res[i] = sum;
	}
}

// Count non-zeros elements inside the matrix
int kernel_matrix_nonzeros(float* mat, int ni, int nj) {
	int i, j, ind;
	int c=0;
	for (i=0; i<ni; ++i) {
		ind = i*nj;
		for (j=0; j<nj; ++j) {
			if (mat[ind + j] != 0) {++c;}
		}
	}
	return c;
}

// Count non-zeros elements per rows inside a matrix
void kernel_matrix_nonzeros_rows(float* mat, int ni, int nj, int* rows, int nrows) {
	int i, j, ind;
	int c = 0;
	for(i=0; i<ni; ++i) {
		ind = i*nj;
		c = 0;
		for (j=0; j<nj; ++j) {
			if (mat[ind + j] != 0) {++c;}
		}
		rows[i] = c;
	}
}

// Compute matrix col sum
void kernel_matrix_sumcol(float* mat, int ni, int nj, float* im, int npix) {
	int i, j, ind;
	for (i=0; i<ni; ++i) {
		ind = i*nj;
		for (j=0; j<nj; ++j) {
			im[j] += mat[ind + j];
		}
	}
}

// Count non-zeros elements inside the matrix
int kernel_vector_nonzeros(float* mat, int ni) {
	int i;
	int c=0;
	for (i=0; i<ni; ++i) {
		if (mat[i] != 0) {++c;}
	}
	return c;
}

// Helper to build H matrix for a low pass filter
void kernel_matrix_lp_H(float* mat, int nk, int nj, int ni, float fc, int order) {
	int i, j, k, step;
	float c, r, size, fi, fj, fk, forder;
	
	forder = (float)order * 2.0f;
	step = nj*ni;
	c = ((float)ni - 1.0f) / 2.0f;
	size = (float)nj - 1.0f;
	for (k=0; k<nk; ++k) {
		for (j=0; j<nj; ++j) {
			for (i=0; i<ni; ++i) {
				fi = (float)i;
				fj = (float)j;
				fk = (float)k;
				r = sqrt((fi-c)*(fi-c) + (fj-c)*(fj-c) + (fk-c)*(fk-c));
				r = r / size;
				r = pow((r / fc), forder);
				r = sqrt(1 + r);
				mat[k*step + i*nj +j] = 1 / r;
			}
		}
	}

}

// Quich Gaussian filter on volume
void kernel_flatvolume_gaussian_filter_3x3x3(float* mat, int nmat, int nk, int nj, int ni) {
	float kernel[] = {4.0f/14.0f, 6.0f/14.0f, 4.0f/14.0f};
	float sum;
	int i, j, k, indi, indk;
	float* res = (float*)calloc(nmat, sizeof(float));
	int step = ni*nj;
	// first on x
	for (k=1; k<(nk-1); ++k) {
		indk = k*step;
		for (i=1; i<(ni-1); ++i) {
			indi = indk + i*nj;
			for (j=1; j<(nj-1); ++j) {
				sum = 0.0f;
				sum += (mat[indi+j-1] * kernel[0]);
				sum += (mat[indi+j] * kernel[1]);
				sum += (mat[indi+j+1] * kernel[2]);
				res[indi+j] = sum;
			}
		}
	}
	// then on y
	for (k=1; k<(nk-1); ++k) {
		indk = k*step;
		for (j=1; i<(nj-1); ++j) {
			for (i=1; i<(ni-1); ++i) {
				sum = 0.0f;
				sum += (mat[indk+(i-1)*nj+j] * kernel[0]);
				sum += (mat[indk+i*nj+j] * kernel[1]);
				sum += (mat[indk+(i+1)*nj+j] * kernel[2]);
				res[indk+i*nj+j] = sum;
			}
		}
	}
	// at the end on z
	for (i=1; i<(ni-1); ++i) {
		indi = i*nj;
		for (j=1; j<(nj-1); ++j) {
			indk = indi+j;
			for (k=1; k<(nk-1); ++k) {
				sum = 0.0f;
				sum += (mat[(k-1)*step+indk] * kernel[0]);
				sum += (mat[k*step+indk] * kernel[1]);
				sum += (mat[(k+1)*step+indk] * kernel[2]);
				res[k*step+indk] = sum;
			}
		}
	}
	// swap result
	memcpy(mat, res, nmat*sizeof(float));
	free(res);
}

// Read a subset of list-mode data set (int data).
void kernel_listmode_open_subset_xyz_int(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1, unsigned short int* z1, int nz1, 
										 unsigned short int* x2, int nx2, unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
										 int n_start, int n_stop, char* basename) {

	// init file
	FILE * pfile_x1;
	FILE * pfile_y1;
	FILE * pfile_z1;
	FILE * pfile_x2;
	FILE * pfile_y2;
	FILE * pfile_z2;
	char namex1 [100];
	char namey1 [100];
	char namez1 [100];
	char namex2 [100];
	char namey2 [100];
	char namez2 [100];
	sprintf(namex1, "%s.x1", basename);
	sprintf(namey1, "%s.y1", basename);
	sprintf(namez1, "%s.z1", basename);
	sprintf(namex2, "%s.x2", basename);
	sprintf(namey2, "%s.y2", basename);
	sprintf(namez2, "%s.z2", basename);
	pfile_x1 = fopen(namex1, "rb");
	pfile_y1 = fopen(namey1, "rb");
	pfile_z1 = fopen(namez1, "rb");
	pfile_x2 = fopen(namex2, "rb");
	pfile_y2 = fopen(namey2, "rb");
	pfile_z2 = fopen(namez2, "rb");
	// position file
	long int pos = n_start * sizeof(unsigned short int);
	fseek(pfile_x1, pos, SEEK_SET);
	fseek(pfile_y1, pos, SEEK_SET);
	fseek(pfile_z1, pos, SEEK_SET);
	fseek(pfile_x2, pos, SEEK_SET);
	fseek(pfile_y2, pos, SEEK_SET);
	fseek(pfile_z2, pos, SEEK_SET);

	// read data
	int i;
	unsigned short int xi1, yi1, zi1, xi2, yi2, zi2;
	int N = n_stop - n_start;
	for (i=0; i<N; ++i) {
		fread(&xi1, 1, sizeof(unsigned short int), pfile_x1);
		fread(&yi1, 1, sizeof(unsigned short int), pfile_y1);
		fread(&zi1, 1, sizeof(unsigned short int), pfile_z1);
		fread(&xi2, 1, sizeof(unsigned short int), pfile_x2);
		fread(&yi2, 1, sizeof(unsigned short int), pfile_y2);
		fread(&zi2, 1, sizeof(unsigned short int), pfile_z2);
		x1[i] = xi1;
		y1[i] = yi1;
		z1[i] = zi1;
		x2[i] = xi2;
		y2[i] = yi2;
		z2[i] = zi2;
	}
	// close files
	fclose(pfile_x1);
	fclose(pfile_y1);
	fclose(pfile_z1);
	fclose(pfile_x2);
	fclose(pfile_y2);
	fclose(pfile_z2);

}

// Read a subset of list-mode data set (float data).
void kernel_listmode_open_subset_xyz_float(float* x1, int nx1, float* y1, int ny1, float* z1, int nz1, 
										   float* x2, int nx2, float* y2, int ny2, float* z2, int nz2,
										   int n_start, int n_stop, char* basename) {

	// init file
	FILE * pfile_x1;
	FILE * pfile_y1;
	FILE * pfile_z1;
	FILE * pfile_x2;
	FILE * pfile_y2;
	FILE * pfile_z2;
	char namex1 [100];
	char namey1 [100];
	char namez1 [100];
	char namex2 [100];
	char namey2 [100];
	char namez2 [100];
	sprintf(namex1, "%s.x1", basename);
	sprintf(namey1, "%s.y1", basename);
	sprintf(namez1, "%s.z1", basename);
	sprintf(namex2, "%s.x2", basename);
	sprintf(namey2, "%s.y2", basename);
	sprintf(namez2, "%s.z2", basename);
	pfile_x1 = fopen(namex1, "rb");
	pfile_y1 = fopen(namey1, "rb");
	pfile_z1 = fopen(namez1, "rb");
	pfile_x2 = fopen(namex2, "rb");
	pfile_y2 = fopen(namey2, "rb");
	pfile_z2 = fopen(namez2, "rb");
	// position file
	long int pos = n_start * sizeof(float);
	fseek(pfile_x1, pos, SEEK_SET);
	fseek(pfile_y1, pos, SEEK_SET);
	fseek(pfile_z1, pos, SEEK_SET);
	fseek(pfile_x2, pos, SEEK_SET);
	fseek(pfile_y2, pos, SEEK_SET);
	fseek(pfile_z2, pos, SEEK_SET);

	// read data
	int i;
	float xf1, yf1, zf1, xf2, yf2, zf2;
	int N = n_stop - n_start;
	for (i=0; i<N; ++i) {
		fread(&xf1, 1, sizeof(float), pfile_x1);
		fread(&yf1, 1, sizeof(float), pfile_y1);
		fread(&zf1, 1, sizeof(float), pfile_z1);
		fread(&xf2, 1, sizeof(float), pfile_x2);
		fread(&yf2, 1, sizeof(float), pfile_y2);
		fread(&zf2, 1, sizeof(float), pfile_z2);
		x1[i] = xf1;
		y1[i] = yf1;
		z1[i] = zf1;
		x2[i] = xf2;
		y2[i] = yf2;
		z2[i] = zf2;
	}
	// close files
	fclose(pfile_x1);
	fclose(pfile_y1);
	fclose(pfile_z1);
	fclose(pfile_x2);
	fclose(pfile_y2);
	fclose(pfile_z2);

}

// Read a subset of list-mode data set (Id of crystals and detectors).
void kernel_listmode_open_subset_ID_int(int* idc1, int n1, int* idd1, int n2, int* idc2, int n3, int* idd2, int n4,
										int n_start, int n_stop, char* name) {

	// init file
	FILE * pfile;
	pfile = fopen(name, "rb");
	// position file
	long int pos = 4 * n_start * sizeof(int);
	fseek(pfile, pos, SEEK_SET);
	// read data
	int i;
	int c1, c2, d1, d2;
	int N = n_stop - n_start;
	for (i=0; i<N; ++i) {
		fread(&c1, 1, sizeof(int), pfile);
		fread(&d1, 1, sizeof(int), pfile);
		fread(&c2, 1, sizeof(int), pfile);
		fread(&d2, 1, sizeof(int), pfile);
		idc1[i] = c1;
		idd1[i] = d1;
		idc2[i] = c2;
		idd2[i] = d2;
	}
	// close files
	fclose(pfile);

}

/**************************************************************
 * Utils
 **************************************************************/

// TO TEST
void toto(char* name) {
	printf("%s\n", name);
}



void kernel_pet3D_IM_DEV_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
							  unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
							  unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
							  int* im, int nim, int wim, int ID) {
	kernel_pet3D_IM_DEV_wrap_cuda(x1, nx1, y1, ny1, z1, nz1, x2, nx2, y2, ny2, z2, nz2, im, nim, wim, ID);
}

void kernel_pet3D_IM_SRM_DDA_ON_iter_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
										  unsigned short int* z1, int nz1,	unsigned short int* x2, int nx2,
										  unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
										  float* im, int nim, float* F, int nf, int wim, int ID) {
	kernel_pet3D_IM_SRM_DDA_ON_iter_wrap_cuda(x1, nx1, y1, ny1, z1, nz1, x2, nx2, y2, ny2, z2, nz2, im, nim, F, nf, wim, ID);
}

void kernel_pet3D_IM_ATT_SRM_DDA_ON_iter_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
											  unsigned short int* z1, int nz1,	unsigned short int* x2, int nx2,
											  unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
											  float* im, int nim, float* F, int nf, float* mumap, int nmu, int wim, int ID) {
	kernel_pet3D_IM_ATT_SRM_DDA_ON_iter_wrap_cuda(x1, nx1, y1, ny1, z1, nz1, x2, nx2, y2, ny2, z2, nz2, im, nim, F, nf, mumap, nmu, wim, ID);
}

/**************************************************************
 * DEVs
 **************************************************************/

#define pi  3.141592653589
void kernel_mip_volume_rendering(float* vol, int nz, int ny, int nx, float* mip, int him, int wim, float alpha, float beta, float scale) {
	// first some var
	float ts = 0.5 * sqrt(nz*nz + nx*nx) + 1;
	float sizeworld = 2 * wim;
	float center_world = sizeworld / 2.0;
	float center_imx = wim / 2.0;
	float center_imy = him / 2.0;
	float padx = (sizeworld-nx) / 2.0;
	float pady = (sizeworld-ny) / 2.0;
	float padz = (sizeworld-nz) / 2.0;
	int step = nx*ny;
	//printf("ts %f size %f center %f imx %f imy %f\n", ts, sizeworld, center_world, center_imx, center_imy);
	int x, y;
	float xw, yw, zw, x1, y1, z1, x2, y2, z2;
	float xd, yd, zd, xmin, ymin, zmin, xmax, ymax, zmax;
	float tmin, tmax, tymin, tymax, tzmin, tzmax, buf;
	float xp1, yp1, zp1, xp2, yp2, zp2;
	int length, lengthy, lengthz, i;
	float xinc, yinc, zinc, maxval, val;

	float ca, sa, cb, sb;
	ca = cos(alpha);
	sa = sin(alpha);
	cb = cos(beta);
	sb = sin(beta);

	for (y=0; y<him; ++y) {
		for (x=0; x<wim; ++x) {
			// init image
			mip[y*wim + x] = 0.0f;
			// origin centre in the world
			xw = x - center_imx;
			yw = y - center_imy;
			zw = -ts;

			// magnefication
			xw = xw * scale;
			yw = yw * scale;
			
			// Rotation 2 axes
			x1 = xw*ca + zw*sa;
			y1 = xw*sb*sa + yw*cb - zw*sb*ca;
			z1 = -xw*sa*cb + yw*sb + zw*cb*ca;
			zw = ts;
			x2 = xw*ca + zw*sa;
			y2 = xw*sb*sa + yw*cb - zw*sb*ca;
			z2 = -xw*sa*cb + yw*sb + zw*cb*ca;
			
			/* One axe
			x1 = xw*cos(alpha) + zw*sin(alpha);
			y1 = yw;
			z1 = -xw*sin(alpha) + zw*cos(alpha);
			zw = ts;
			x2 = xw*cos(alpha) + zw*sin(alpha);
			y2 = yw;
			z2 = -xw*sin(alpha) + zw*cos(alpha);
			*/

			//printf("%f %f %f\n", x1, y1, z1);
			//printf("%f %f %f\n", x2, y2, z2);
			// change origin to raycasting
			x1 += center_world;
			y1 += center_world;
			z1 += center_world;
			x2 += center_world;
			y2 += center_world;
			z2 += center_world;
			// define box and ray direction
			xmin = padx;
			xmax = padx+float(nx);
			ymin = pady;
			ymax = pady+float(ny);
			zmin = padz;
			zmax = padz+float(nz);
			// Rayscasting Smith's algorithm ray-box AABB intersection
			xd = x2 - x1;
			yd = y2 - y1;
			zd = z2 - z1;
			tmin = -1e9f;
			tmax = 1e9f;
			// on x
			if (xd != 0.0f) {
				tmin = (xmin - x1) / xd;
				tmax = (xmax - x1) / xd;
				if (tmin > tmax) {
					buf = tmin;
					tmin = tmax;
					tmax = buf;
				}
			}
			// on y
			if (yd != 0.0f) {
				tymin = (ymin - y1) / yd;
				tymax = (ymax - y1) / yd;
				if (tymin > tymax) {
					buf = tymin;
					tymin = tymax;
					tymax = buf;
				}
				if (tymin > tmin) {tmin = tymin;}
				if (tymax < tmax) {tmax = tymax;}
			}
			// on z
			if (zd != 0.0f) {
				tzmin = (zmin - z1) / zd;
				tzmax = (zmax - z1) / zd;
				if (tzmin > tzmax) {
					buf = tzmin;
					tzmin = tzmax;
					tzmax = buf;
				}
				if (tzmin > tmin) {tmin = tzmin;}
				if (tzmax < tmax) {tmax = tzmax;}
			}
			// compute points
			xp1 = x1 + xd * tmin;
			yp1 = y1 + yd * tmin;
			zp1 = z1 + zd * tmin;
			xp2 = x1 + xd * tmax;
			yp2 = y1 + yd * tmax;
			zp2 = z1 + zd * tmax;
			//printf("p1 %f %f %f - p2 %f %f %f\n", xp1, yp1, zp1, xp2, yp2, zp2);
			// check point p1
			if (xp1 >= xmin && xp1 <= xmax) {
				if (yp1 >= ymin && yp1 <= ymax) {
					if (zp1 >= zmin && zp1 <= zmax) {
						xp1 -= padx;
						yp1 -= pady;
						zp1 -= padz;
						if (int(xp1+0.5) == nx) {xp1 = nx-1.0f;}
						if (int(yp1+0.5) == ny) {yp1 = ny-1.0f;}
						if (int(zp1+0.5) == nz) {zp1 = nz-1.0f;}
					} else {continue;}
				} else {continue;}
			} else {continue;}
			// check point p2
			if (xp2 >= xmin && xp2 <= xmax) {
				if (yp2 >= ymin && yp2 <= ymax) {
					if (zp2 >= zmin && zp2 <= zmax) {
						xp2 -= padx;
						yp2 -= pady;
						zp2 -= padz;
						if (int(xp2+0.5) == nx) {xp2 = nx-1.0f;}
						if (int(yp2+0.5) == ny) {yp2 = ny-1.0f;}
						if (int(zp2+0.5) == nz) {zp2 = nz-1.0f;}
					} else {continue;}
				} else {continue;}
			} else {continue;}

			//printf("e %f %f %f    s %f %f %f\n", xp1, yp1, zp1, xp2, yp2, zp2);

			// walk the ray to choose the max intensity with the DDA algorithm
			step = nx * ny;
			length = abs(xp2 - xp1);
			lengthy = abs(yp2 - yp1);
			lengthz = abs(zp2 - zp1);
			if (lengthy > length) {length = lengthy;}
			if (lengthz > length) {length = lengthz;}
			
			xinc = (xp2 - xp1) / (float)length;
			yinc = (yp2 - yp1) / (float)length;
			zinc = (zp2 - zp1) / (float)length;
			xp1 += 0.5;
			yp1 += 0.5;
			zp1 += 0.5;
			maxval = 0.0f;
			for (i=0; i<=length; ++i) {
				val = vol[(int)zp1*step + (int)yp1*nx + (int)xp1];
				if (val > maxval) {maxval = val;}
				xp1 += xinc;
				yp1 += yinc;
				zp1 += zinc;
			}
			
			// Assign new value
			mip[y*wim + x] = maxval;
			
		} // loop j
	} // loop i


}

#undef pi
