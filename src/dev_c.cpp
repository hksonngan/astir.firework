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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <dev_c.h>

/********************************************************************************
 * Dev file
 ********************************************************************************/

// here, put your code!

/********************************************************************************
 * Utils
 ********************************************************************************/
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876
float rnd_park_miller(int *seed) {
	int k;
	float ans;
	*seed ^= MASK;
	k = (*seed)/IQ;
	*seed = IA*(*seed-k*IQ)-IR*k;
	if (*seed < 0) {*seed += IM;}
	ans = AM*(*seed);
	*seed ^= MASK;
	return ans;
}
#undef IA
#undef IM
#undef AM
#undef IQ
#undef IR
#undef MASK

/********************************************************************************
 * 3D ray-traversal test                                   JB 2011-02-03 15:04:14
 ********************************************************************************/
void dev_siddon_3D(float* vol, int nz, int ny, int nx, int nlines) {
	int p;
	int u, v, w;
	int i, j, k;
	int ei, ej, ek;
	int stepi, stepj, stepk;
	int oldi, oldj, oldk;
	float x0, y0, z0;
	float xe, ye, ze;
	float stepx, stepy, stepz;
	float axstart, aystart, azstart;
	float runx, runy, runz, pq, oldv, newv, val;
	float eps = 1.0e-10f;
	float totv;
	int jump = ny*nx;
	int seed = 100;
	for (p=0; p<nlines; ++p) {
		x0 = rnd_park_miller(&seed)*nx;
		y0 = rnd_park_miller(&seed)*ny;
		z0 = rnd_park_miller(&seed)*nz;
		
		xe = rnd_park_miller(&seed)*nx;
		ye = rnd_park_miller(&seed)*ny;
		ze = rnd_park_miller(&seed)*nz;

		ei = int(x0);
		ej = int(y0);
		ek = int(z0);

		if ((xe-x0) > 0) {stepi = 1; u = ei+1;}
		if ((xe-x0) < 0) {stepi = -1; u = ei;}
		if ((xe-x0) == 0) {stepi = 0; u = ei; xe = eps;}

		if ((ye-y0) > 0) {stepj = 1; v = ej+1;}
		if ((ye-y0) < 0) {stepj = -1; v = ej;}
		if ((ye-y0) == 0) {stepj = 0; v = ej; ye = eps;}

		if ((ze-z0) > 0) {stepk = 1; w = ek+1;}
		if ((ze-z0) < 0) {stepk = -1; w = ek;}
		if ((ze-z0) == 0) {stepk = 0; w = ek; ze = eps;}

		axstart = (u - x0) / (xe - x0);
		aystart = (v - y0) / (ye - y0);
		azstart = (w - z0) / (ze - z0);

		pq = sqrt((x0-xe)*(x0-xe)+(y0-ye)*(y0-ye)+(z0-ze)*(z0-ze));
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		oldv = runx;
		if (runy < oldv) {oldv = runy;}
		if (runz < oldv) {oldv = runz;}
		stepx = fabs((pq / (xe-x0)));
		stepy = fabs((pq / (ye-y0)));
		stepz = fabs((pq / (ze-z0)));
		i = ei;
		j = ej;
		k = ek;

		if (runx == oldv) {runx += stepx; i += stepi;}
		if (runy == oldv) {runy += stepy; j += stepj;}
		if (runz == oldv) {runz += stepz; k += stepk;}

		vol[ek*jump + ej*nx + ei] += oldv;
		totv = 0.0f;
		oldi = i;
		oldj = j;
		oldk = k;

		while (oldv < pq) {
			newv = runx;
			if (runy < newv) {newv=runy;}
			if (runz < newv) {newv=runz;}
			val = newv - oldv;
			vol[k*jump + j*nx + i] += val;

			totv += val;
			oldv = newv;
			oldi = i;
			oldj = j;
			oldk = k;
			if (runx==newv) {i += stepi; runx += stepx;}
			if (runy==newv) {j += stepj; runy += stepy;}
			if (runz==newv) {k += stepk; runz += stepz;}
		}

		// last point (correct value)
		vol[oldk*jump + oldj*nx + oldi] += (pq - totv);

	} // for p
}

void dev_amanatides_3D(float* vol, int nz, int ny, int nx,
					   float* X0, int nx0, float* Y0, int ny0, float* Z0, int nz0,
					   float* Xe, int nxe, float* Ye, int nye, float* Ze, int nze) {
	int p;
	int u, v, w;
	int i, j, k;
	int ei, ej, ek;
	int stepi, stepj, stepk;
	int oldi, oldj, oldk;
	float x0, y0, z0;
	float xe, ye, ze;
	float stepx, stepy, stepz;
	float axstart, aystart, azstart;
	float runx, runy, runz, pq, d, oldd;
	float eps = 1.0e-10f;
	int jump = ny*nx;
	for (p=0; p<nx0; ++p) {
		x0 = X0[p];
		y0 = Y0[p];
		z0 = Z0[p];
		xe = Xe[p];
		ye = Ye[p];
		ze = Ze[p];

		ei = int(x0);
		ej = int(y0);
		ek = int(z0);

		if ((xe-x0) > 0) {stepi = 1; u = ei+1;}
		if ((xe-x0) < 0) {stepi = -1; u = ei;}
		if ((xe-x0) == 0) {stepi = 0; u = ei; xe = eps;}

		if ((ye-y0) > 0) {stepj = 1; v = ej+1;}
		if ((ye-y0) < 0) {stepj = -1; v = ej;}
		if ((ye-y0) == 0) {stepj = 0; v = ej; ye = eps;}

		if ((ze-z0) > 0) {stepk = 1; w = ek+1;}
		if ((ze-z0) < 0) {stepk = -1; w = ek;}
		if ((ze-z0) == 0) {stepk = 0; w = ek; ze = eps;}

		axstart = (u - x0) / (xe - x0);
		aystart = (v - y0) / (ye - y0);
		azstart = (w - z0) / (ze - z0);

		pq = sqrt((x0-xe)*(x0-xe)+(y0-ye)*(y0-ye)+(z0-ze)*(z0-ze));
		runx = axstart * pq;
		runy = aystart * pq;
		runz = azstart * pq;
		stepx = fabs((pq / (xe-x0)));
		stepy = fabs((pq / (ye-y0)));
		stepz = fabs((pq / (ze-z0)));
		i = ei;
		j = ej;
		k = ek;
		oldd = runx;
		if (runy < oldd) {oldd = runy;}
		if (runz < oldd) {oldd = runz;}
		
		vol[k*jump + j*nx + i] += oldd;
		d = 0.0f;
		
		while (d < pq) {
			if (runx < runy) {
				if (runx < runz) {i += stepi; runx += stepx;}
				else {k += stepk; runz += stepz;}
			} else {
				if (runy < runz) {j += stepj; runy += stepy;}
				else {k += stepk; runz += stepz;}
			}
			d = runx;
			if (runy < d) {d = runy;}
			if (runz < d) {d = runz;}
			vol[k*jump + j*nx + i] += (d-oldd);
			oldd = d;
		}
		// last point (correct value)
		vol[k*jump + j*nx + i] += (pq - d);

	} // for p
}

void dev_raypro_3D(float* vol, int nz, int ny, int nx,
				   float* X0, int nx0, float* Y0, int ny0, float* Z0, int nz0,
				   float* DX, int ndx, float* DY, int ndy, float* DZ, int ndz,
				   float* D, int nd) {
	float xi, yi, zi;
	float x0, y0, z0;
	float dx, dy, dz, idx, idy, idz;
	float dbx, dby, dbz;
	float sdx, sdy, sdz;
	float t, tn, tot_t, d;
	int xp, yp, zp;
	int bx, by, bz;
	int obx, oby, obz;
	int jump = nx*ny;
	int p;
	float eps = 1.0e-5f;
	
	for (p=0; p<nx0; ++p) {
		x0 = X0[p];
		y0 = Y0[p];
		z0 = Z0[p];
		dx = DX[p];
		dy = DY[p];
		dz = DZ[p];
		d = D[p];

		if (dx==0) {dx=eps;}
		if (dy==0) {dy=eps;}
		if (dz==0) {dz=eps;}
		idx = 1.0f / dx;
		idy = 1.0f / dy;
		idz = 1.0f / dz;
		
		// sign
		sdx = dx / fabs(dx);
		sdy = dy / fabs(dy);
		sdz = dz / fabs(dz);

		sdx = copysign(1.0f, dx);
		sdy = copysign(1.0f, dy);
		sdz = copysign(1.0f, dz);
		
		// compute
		dbx = (0.5f+eps)*sdx + (0.5f-eps);
		dby = (0.5f+eps)*sdy + (0.5f-eps);
		dbz = (0.5f+eps)*sdz + (0.5f-eps);
		
		//dbx = (dx > 0) - (dx < 0) * eps;
		//dby = (dy > 0) - (dy < 0) * eps;
		//dbz = (dz > 0) - (dz < 0) * eps;
		
        // first point
		bx = int(x0+dbx);
		by = int(y0+dby);
		bz = int(z0+dbz);
		obx = bx;
		oby = by;
		obz = bz;

		t = (bx-x0)*idx; // tx
		tn = (by-y0)*idy; // ty		
		if (tn < t) {t=tn;}
		tn = (bz-z0)*idz; // tz
		if (tn < t) {t=tn;}

		xi = x0 + (dx*t);
		yi = y0 + (dy*t);
		zi = z0 + (dz*t);
		
		tn = 1.0f + int(xi) - xi;
		xi += (tn * (tn < eps));
		tn = 1.0f + int(yi) - yi;
		yi += (tn * (tn < eps));
		tn = 1.0f + int(zi) - zi;
		zi += (tn * (tn < eps));
		
		tot_t = t;
		xp = int(x0);
		yp = int(y0);
		zp = int(z0);
		vol[zp*jump + yp*nx + xp] += t;

		while (tot_t < d) {
			bx = int(xi+dbx);
			by = int(yi+dby);
			bz = int(zi+dbz);

			t = (bx - xi) * idx; // tx
			tn = (by - yi) * idy; // ty
			if (tn < t) {t=tn;}
			tn = (bz - zi) * idz; // tz
			if (tn < t) {t=tn;}

			xi = xi + (dx*t);
			yi = yi + (dy*t);
			zi = zi + (dz*t);

			tn = 1.0f + int(xi) - xi;
			xi += (tn * (tn < eps));
			tn = 1.0f + int(yi) - yi;
			yi += (tn * (tn < eps));
			tn = 1.0f + int(zi) - zi;
			zi += (tn * (tn < eps));

			tot_t += t;
			xp += (bx - obx);
			yp += (by - oby);
			zp += (bz - obz);
			obx = bx;
			oby = by;
			obz = bz;

			vol[zp*jump + yp*nx + xp] += t;
			
		}

		// retrieve correct t value due to overshoot
		vol[zp*jump + yp*nx + xp] += (d-tot_t);
		//printf("ite %i\n", watchdog);
			
	} // for p

}

void dev_mc_distribution(float* dist, int nz, int ny, int nx,
						 float* res, int nrz, int nrx, int nry, int N) {
	int i, j;
	int nb = nx*ny*nz;
	float tot = 0;
	float rnd;

	i=0;
	while (i<nb) {tot += dist[i]; ++i;}
	tot = 1.0f / tot;

	i=0;
	while (i<nb) {dist[i] *= tot; ++i;}
	
	i=1;
	while (i<nb) {dist[i] += (dist[i-1]);	++i;}

	i=0;
	while (i<N) {
		rnd = (float)rand() / (float)(RAND_MAX+1.0f);
		j = 0;
		while (dist[j] < rnd) {++j;}
		res[j] += 1.0f;
		++i;
	}
}
