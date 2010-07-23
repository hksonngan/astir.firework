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
#include <math.h>
#include <omp.h>
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
		a = (float)id_detector1[n] * (-twopi / (float)nd);
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
		a = (float)id_detector2[n] * (-twopi / (float)nd);
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

// SRM Raytracing (transversal algorithm), Compute entry and exit point on SRM of the ray
void kernel_pet2D_SRM_entryexit(float* px, int npx, float* py, int npy, float* qx, int nqx, float* qy, int nqy, int b, int res, int srmsize, int* enable, int nenable) {
	float divx, divy, fsrmsize;
	float axn, ax0, ayn, ay0;
	float amin, amax, buf1, buf2;
	float x1, y1, x2, y2;
	float pxi, pyi, qxi, qyi;
	int xi1, yi1, xi2, yi2;
	int i;
		
	b = (float)b;
	res = (float)res;
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
		axn = (fsrmsize * res + b - qxi) / divx;
		ax0 = (b - qxi) / divx;
		ayn = (fsrmsize * res + b - qyi) / divy;
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

		x1 = (qxi + amax * (pxi - qxi) - b) / res;
		y1 = (qyi + amax * (pyi - qyi) - b) / res;
		x2 = (qxi + amin * (pxi - qxi) - b) / res;
		y2 = (qyi + amin * (pyi - qyi) - b) / res;

		// format
		xi1 = (int)x1;
		if (xi1 == srmsize) {xi1 = srmsize-1;}
		yi1 = (int)y1;
		if (yi1 == srmsize) {yi1 = srmsize-1;}
		xi2 = (int)x2;
		if (xi2 == srmsize) {xi2 = srmsize-1;}
		yi2 = (int)y2;
		if (yi2 == srmsize) {yi2 = srmsize-1;}
		// check if ray through the image
		enable[i] = 1;
		if (xi1 < 0 || xi1 > srmsize || yi1 < 0 || yi1 > srmsize) {enable[i] = 0;}
		if (xi2 < 0 || xi2 > srmsize || yi2 < 0 || yi2 > srmsize) {enable[i] = 0;}
		// check if the ray is > 0
		if (xi1 == xi2 && yi1 == yi2) {enable[i] = 0;}
		px[i] = xi1;
		py[i] = yi1;
		qx[i] = xi2;
		qy[i] = yi2;
	}
}

// Cleanning LORs outside of ROi based on SRM entry-exit point calculation
void kernel_pet2D_SRM_clean_entryexit(int* enable, int ne, float* x1, int nx1, float* y1, int ny1, float* x2, int nx2, float* y2, int ny2,
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
		val  = 1 / flength;
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
void kernel_pet2D_SRM_SIDDON(float* SRM, int wy, int wx, float* X1, int nx1, float* Y1, int ny1, float* X2, int nx2, float* Y2, int ny2, int res, int b, int matsize) {
	int n, LOR_ind;
	float tx, ty, px, qx, py, qy;
	int ei, ej, u, v, i, j;
	int stepi, stepj;
	float divx, divy, runx, runy, oldv, newv, val;
	float axstart, aystart, astart, pq, stepx, stepy, startl;
	for (n=0; n<nx1; ++n) {
		LOR_ind = n * wx;

		px = X2[n];
		py = Y2[n];
		qx = X1[n];
		qy = Y1[n];
		tx = (px-qx) * 0.5 + qx;
		ty = (py-qy) * 0.5 + qy;
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
		oldv = 0.0f;
		if (runx < runy) {oldv = runx;}
		while (1) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			val = fabs(newv - oldv);
			if (val > 10.0) {val = 1.0;}
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
			if (i>=(matsize-1) || j>=(matsize-1) || i<=0 || j<=0) {break;}
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
		
		SRM[LOR_ind + ej * matsize + ei] = fabs(newv - oldv);
		oldv = 0.0f;
		while (1) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			val = fabs(newv - oldv);
			if (val > 10.0) {val = 1.0;}
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
			if (i>=(matsize-1) || j>=(matsize-1) || i<=0 || j<=0) {break;}
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
	float divx, divy, runx, runy, oldv, newv;
	float axstart, aystart, astart, pq, stepx, stepy, startl;
	for (n=0; n<nx1; ++n) {
		px = X2[n];
		py = Y2[n];
		qx = X1[n];
		qy = Y1[n];
		tx = (px-qx) * 0.5 + qx;
		ty = (py-qy) * 0.5 + qy;
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
		while (1) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			mat[j * wx + i] += fabs(newv-oldv);
			if (i>=(matsize-1) || j>=(matsize-1) || i<=0 || j<=0) {break;}
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
		mat[ej * wx + ei] += fabs(newv - oldv);
		oldv = startl;
		while (1) {
			newv = runy;
			if (runx < runy) {newv = runx;}
			mat[j * wx + i] += fabs(newv - oldv);
			if (i>=(matsize-1) || j>=(matsize-1) || i<=0 || j<=0) {break;}
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

// EM-ML algorithm, only one iteration (list-mode)
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

// EM-ML algorithm with sparse matrix (COO), only one iteration (list-mode)
void kernel_pet2D_LM_EMML_COO_iter(float* SRMvals, int nvals, int* SRMrows, int nrows, int* SRMcols, int ncols, float* S, int nbs, float* im, int npix, int nevents) {
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
	printf("Q %f %f %f\n", Q[0], Q[1], Q[2]);
	// Sparse matrix operation F = SRM^T * Q
	for (i=0; i<nvals; ++i) {
		F[SRMcols[i]] += (SRMvals[i] / Q[SRMrows[i]]);
	}
	printf("F %f %f %f\n", F[0], F[1], F[2]);

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
	printf("Q %f %f %f\n", Q[0], Q[1], Q[2]);
	// Sparse matrix operation F = SRM^T * Q
	for (i=0; i<nivals; ++i) {
		ind = i * njvals;
		vcol = SRMcols[ind];
		j = 0;
		while (vcol != -1) {
			F[vcol] += (SRMvals[ind+j] * Q[i]);
			++j;
			vcol = SRMcols[ind+j];
		}
	}
	printf("F %f %f %f\n", F[0], F[1], F[2]);
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
