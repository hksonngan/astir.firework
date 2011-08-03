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

#include "dev.h"

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


void printf_vec(float* A, int nA) {
	int i=0;
	while (i<nA) {printf("%f ", A[i]); ++i;}
	printf("\n");
}

void printi_vec(int* A, int nA) {
	int i=0;
	while (i<nA) {printf("%i ", A[i]); ++i;}
	printf("\n");
}


void dev_MSPS_build(float* org_act, int nact, int* ind, int nind) {
	//printf("ok\n");

	int lambda = 3;
	int level = 10;
	int totelt = int(nact*1.5) + level;

	int c, k, i;
	double sum=0.0;

	float* MS = (float*)malloc(totelt*sizeof(float));
	int* indk = (int*)malloc(level*sizeof(int));
	int* nk = (int*)malloc(level*sizeof(int));
	memset(MS, 0, totelt);

	// Double precision is require otherwise the sum of the normalize vector != 1
	double* act = (double*)malloc(nact*sizeof(double));
	double s=0.0;
	
	// normamize the org_act
	i=0; while (i<nact) {s += org_act[i]; ++i;}
	i=0; while (i<nact) {act[i] = org_act[i] / s; ++i;}
	// accumulate values
	i=1; while (i<nact) {act[i] += act[i-1]; ++i;}

	// copy the data to the first level
	i=0; while (i<nact) {MS[i] = (float)act[i]; ++i;}
	indk[0] = 0;
	nk[0] = nact;
	
	// build every level
	int ilevel;
	for (ilevel = 1; ilevel < level; ++ilevel) {
		nk[ilevel] = int(nk[ilevel-1] / lambda) + 1;
		//if (fmod(nk[ilevel-1], lambda) >= 1) {++nk[ilevel];}
		indk[ilevel] = indk[ilevel-1] + nk[ilevel-1];
		k = indk[ilevel];
	
		c=0; i=0; sum=0.0f;
		for (i=indk[ilevel-1]; i<indk[ilevel]; ++i) {
			sum += MS[i];

			++c;
			if (c==lambda) {
				MS[k] = sum / float(lambda);
				sum = 0.0f;
				c = 0;
				++k;
			}
		}
		//if (k != indk[1]) {MS[k] = 1.0f;}
		MS[k] = 1.0f;
	}

	/*
	//printf_vec(MS, nk[0]+nk[1]+nk[2]);

	//printf_vec(MS, nk[0]+nk[1]+nk[2]);
	//printi_vec(nk, level);
	//printi_vec(indk, level);
	for (i=0; i<level; ++i) {
		printf("level %i : [%f - %f]    %i elts\n", i, MS[indk[i]], MS[indk[i]+nk[i]-1], nk[i]);
	}

	printf("[");
	for (i=0; i<nk[level-1]; ++i) {printf("%f ", MS[indk[level-1] + i]);} 
	printf("]\n");
	*/
	// Export
	FILE * pfile = fopen("msv_activity.bin", "wb");
	fwrite(MS, sizeof(float), totelt, pfile);
	fclose(pfile);
	pfile = fopen("msi_activity.bin", "wb");
	fwrite(ind, sizeof(int), nind, pfile);
	fclose(pfile);
	pfile = fopen("nk_activity.bin", "wb");
	fwrite(nk, sizeof(int), level, pfile);
	fclose(pfile);
	pfile = fopen("indk_activity.bin", "wb");
	fwrite(indk, sizeof(int), level, pfile);
	fclose(pfile);
	

	//free(MS);
	//free(indk);
	//free(nk);


}

void dev_MSPS_naive(float* act, int nact, int* indact, int inact,
					float* X, int sx, float* Y, int sy, float* Z, int sz,
					int* step, int nstep,
					int npoint, int seed, int nz, int ny, int nx) {

	int n = 0;
	float rnd;
	float jump = ny*nx;
	float ijump = 1.0f / jump;
	float inx = 1.0f / float(nx);
	float ind, x, y, z;
	int istep, stepmem;
	srand(seed);
	while (n<npoint) {
		rnd = (float)rand() / (float)(RAND_MAX+1.0f);

		istep = int(rnd * nact);
		
		stepmem = istep;
		
		//istep = 0;
		//while (act[istep] < rnd) {++istep;}
		//ind = float(indact[istep]);

		//printf("act_istep %f rnd %f\n", act[istep], rnd);

		
		if (act[istep] < rnd) {
			while (act[istep] < rnd) {++istep;}
		} else {
			while (act[istep] > rnd) {--istep;}
			++istep; // correct undershoot
		}

		//printf("act_istep %f rnd %f\n", act[istep], rnd);
		
		ind = float(indact[istep]);
		
		z = floor(ind * ijump);
		ind -= (z * jump);
		y = floor(ind * inx);
		x = ind - y*nx;
	
		// random position inside voxel
		x += ((float)rand() / (float)(RAND_MAX+1.0f));
		y += ((float)rand() / (float)(RAND_MAX+1.0f));
		z += ((float)rand() / (float)(RAND_MAX+1.0f));
		step[n] = abs(stepmem - istep);

		X[n] = x;
		Y[n] = y;
		Z[n] = z;
	
		++n;
	}

}

void dev_RAIM_gen(float* act, int nact, int* indact, int nindact,
				  float* X, int sx, float* Y, int sy, float* Z, int sz,
				  int* step, int nstep, int seed, int nz, int ny, int nx) {
	srand(seed);
	int n = 0;
	int pos = 0;
	float ind;
	float rnd;
	int istep = 0;
	int jump = ny*nx;
	int x, y, z;
	float eps = 1e-3f;
	while (n<sx) {
		rnd = (float)rand() / (float)(RAND_MAX);
		pos = int(rnd * nact);
		//pos=0;
		istep = 0;
		if (act[pos] < rnd) {
			while (act[pos] < rnd && pos < nact) {++pos; ++istep;}
		} else {
			while (act[pos] > rnd && pos >= 0) {--pos; ++istep;}
			++pos; // correct undershoot
		}

		ind = float(indact[pos]);
		z = floor(ind / float(jump));
		ind -= (z * jump);
		y = floor(ind / float(nx));
		x = ind - y*nx;
	
		// random position inside voxel
		x += ((float)rand() / (float)(RAND_MAX));
		y += ((float)rand() / (float)(RAND_MAX));
		z += ((float)rand() / (float)(RAND_MAX));

		if (x >= nx) {x -= eps;}
		if (y >= ny) {y -= eps;}
		if (z >= nz) {z -= eps;}
	
		X[n] = x;
		Y[n] = y;
		Z[n] = z;

		step[n] = istep;
	
		++n;
	}

}


// Voxelized source generation - Multi-Scale Propagation Search (MSPS)
void dev_MSPS_gen(float* msv, int nmsv, int* msi, int nmsi, int* nk, int nnk, int* indk, int nindk,
				  float* X, int sx, float* Y, int sy, float* Z, int sz, int* step, int nstep,
				  int npoint, int seed, int nz, int ny, int nx) {
				  
				  
	int level = 10;
	int lambda = 3;
	int half_lambda = lambda / 2;
	int i, jloc, jglb;
	float x, y, z, ind, rnd;
	float jump = ny*nx;
	float ijump = 1.0f / jump;
	float inx = 1.0f / float(nx);
	int istep = 0;
	int bormin, bormax;
	float eps = 1e-3f;
	srand(seed);

	int maxind = nz*ny*nx;

	int n = 0;
	while (n < npoint) {
		istep = 0;

		/*
		if (n==307) {
	printf("======= point %i ===========\n", n);
	// display
	for (i=0; i<level; ++i) {
		printf("level %i : [%f - %f]    %i elts   indk %i\n", i, msv[indk[i]], msv[indk[i]+nk[i]-1], nk[i], indk[i]);
	}
	printf("[");
	for (i=0; i<nk[level-1]; ++i) {printf("%f ", msv[indk[level-1] + i]);} 
	printf("]\n");
		}
		*/

	// select a random number
	rnd = (float)rand() / (float)(RAND_MAX);
	//if (n==307) {printf("random number: %f\n", rnd);}

	// first position esimation
	jloc = int(rnd * nk[level-1]);
	if (jloc >= nk[level-1]) {jloc = nk[level-1] - 1;}
	jglb = jloc + indk[level-1];
	/*
	if (n==307) {
	printf("random %f\n", rnd);
	printf("Level %i\n", level-1);
	printf("   init loc %i glb %i val %f\n", jloc, jglb, msv[jglb]);
	}*/

	// search
	bormin = indk[level-1];
	bormax = indk[level-1] + nk[level-1];
	if (msv[jglb] < rnd) {
		while (msv[jglb] < rnd && jglb < bormax) {++jglb; ++istep;}
	} else {
		while (msv[jglb] > rnd && jglb >= bormin) {--jglb; ++istep;}
		++jglb; // correct undershoot
	}
	jloc = jglb-indk[level-1];
	//if (n==307) {
	//	printf("   search loc %i glb %i val %f  - step %i\n", jloc, jglb, msv[jglb], istep);}

	i = level-2;
	while (i>=0) {
		//if (n==307) {printf("Level %i\n", i);}
		// propagation
		jloc = lambda * jloc + half_lambda;
		// check boundary
		if (jloc >= nk[i]) {jloc = nk[i]-1;}
		if (jloc < 0) {jloc = 0;}
		
		jglb = jloc+indk[i];
		//if (n==307) {printf("   propa loc %i glb %i val %f\n", jloc, jglb, msv[jglb]);}
		// search
		bormin = indk[i];
		bormax = indk[i] + nk[i];
		if (msv[jglb] < rnd) {
			while (msv[jglb] < rnd && jglb < bormax) {++jglb; ++istep;}
		} else {
			while (msv[jglb] > rnd && jglb >= bormin ) {--jglb; ++istep;}
			++jglb; // correct undershoot
		}
		jloc = jglb-indk[i];
		//if (n==307) {printf("   search loc %i glb %i val %f  - step %i\n", jloc, jglb, msv[jglb-1], istep);}
		--i;
	}
	
	//printf("\nFinal search in %i steps\n\n", step);
	/*
	int jj = 10000;
	printf("level 1: %f\n", msv[jj+indk[1]]);
	printf("level 0:\n");
	printf_vec(&msv[lambda*jj + half_lambda + indk[0] - 1], 10);
	*/

	// convert ID
	ind = float(msi[jloc]);
	if (ind >= maxind) {printf("ERROR: ind %f >= maxind %i\n", ind, maxind);}
	z = floor(ind * ijump);

	ind -= (z * jump);
	y = floor(ind * inx);
	x = ind - y*nx;
	
	// random position inside voxel
	x += ((float)rand() / (float)(RAND_MAX));
	y += ((float)rand() / (float)(RAND_MAX));
	z += ((float)rand() / (float)(RAND_MAX));

	if (x >= nx) {x -= eps;}
	if (y >= ny) {y -= eps;}
	if (z >= nz) {z -= eps;}
	
	X[n] = x;
	Y[n] = y;
	Z[n] = z;

	step[n] = istep;
	
	++n;
	} // while
}

#include <assert.h>
void dev_MSPS_acc(int* im, int nz, int ny, int nx,
				  float* x, int sx, float* y, int sy, float* z, int sz) {
	int i=0;
	int jump = nx*ny;
	int maxi = nz*ny*nx;
	int ind;

	while (i<sx) {
		ind = int(z[i])*jump + int(y[i])*nx + int(x[i]);
		assert (ind < maxi);
		im[ind] += 1;
		++i;
	}
}

void dev_im_acc(float* im, int ny, int nx, float* x, int sx, float* y, int sy) {
	int i=0;
	int ind;
	int xi, yi;
	while (i<sx) {
		xi = int(x[i]);
		yi = int(y[i]);
		if (xi >= 0.0f && xi < nx && yi >= 0.0f && yi < ny) {
				ind = yi*nx + xi;
				im[ind] += 1.0f;
		}
		++i;
	}
}


/***********************************************
 * Raytracer to Emanuelle BRARD - AMELL
 *         2011-03-16 10:33:39
 ***********************************************/

int dev_AMELL(int* voxel_ind, int nvox, float* voxel_val, int nvox2, int dimx, int dimy, int dimz,
			  float x1, float y1, float z1,
			  float x2, float y2, float z2) {

	int ex, ey, ez;
	int ix, iy, iz;
	int stepi_x, stepi_y, stepi_z;
	int ux, uy, uz;
	float start_x, start_y, start_z;
	float stept_x, stept_y, stept_z;
	float run_x, run_y, run_z;
	float pq, oldv, val, totv;
	float eps = 1.0e-5f;
	int pos = 0;
	int jump = dimx * dimy;

	ex = int(x1);
	ey = int(y1);
	ez = int(z1);

	pq = sqrtf((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2));

	if ((x2-x1) > 0) {stepi_x = 1; ux = ex + 1;}
	if ((x2-x1) < 0) {stepi_x = -1; ux = ex;}
	if ((x2-x1) == 0) {stepi_x = 0; ux = ex;
		if ((x1-eps) >= 0) {x1 -= eps;}
		else {x2 += eps;}
	}

	if ((y2-y1) > 0) {stepi_y = 1; uy = ey + 1;}
	if ((y2-y1) < 0) {stepi_y = -1; uy = ey;}
	if ((y2-y1) == 0) {stepi_y = 0; uy = ey;
		if ((y1-eps) >= 0) {y1 -= eps;}
		else {y2 += eps;}
	}

	if ((z2-z1) > 0) {stepi_z = 1; uz = ez + 1;}
	if ((z2-z1) < 0) {stepi_z = -1; uz = ez;}
	if ((z2-z1) == 0) {stepi_z = 0; uz = ez;
		if ((z1-eps) >= 0) {z1 -= eps;}
		else {z2 += eps;}
	}
	
	run_x = pq * (ux - x1) / (x2 - x1);
	run_y = pq * (uy - y1) / (y2 - y1);
	run_z = pq * (uz - z1) / (z2 - z1);

	stept_x = fabsf((pq / (x2 - x1)));
	stept_y = fabsf((pq / (y2 - y1)));
	stept_z = fabsf((pq / (z2 - z1)));
	ix = ex;
	iy = ey;
	iz = ez;
	
	oldv = run_x;
	if (run_y < oldv) {oldv = run_y;}
	if (run_z < oldv) {oldv = run_z;}
	
	voxel_val[pos] = oldv;
	voxel_ind[pos] = ez*jump + ey*dimx + ex;
	++pos;
		
	totv = 0.0f;
	while (totv < pq) {
		if (run_x < run_y) {
			if (run_x < run_z) {ix += stepi_x; run_x += stept_x;}
			else {iz += stepi_z; run_z += stept_z;}
		} else {
			if (run_y < run_z) {iy += stepi_y; run_y += stept_y;}
			else {iz += stepi_z; run_z += stept_z;}
		}
		totv = run_x;
		if (run_y < totv) {totv=run_y;}
		if (run_z < totv) {totv=run_z;}

		val = totv - oldv;
		if (val != 0.0f) {
			voxel_val[pos] = val;
			voxel_ind[pos] = iz*jump + iy*dimx + ix;
			oldv = totv;
			++pos;
		}
	}

	voxel_val[pos-1] += (pq - totv);

	return pos;
		
}


/********************************************************  
 * Apply new direction to the particle (use quaternion) *
 *                                                      *
 ********************************************************/

typedef struct {
	float x, y, z, w;
} point;

// Hamilton multiplication (quaternion)
point quat_mul(point p, point q) {
	point res;
	res.x = p.x*q.x - p.y*q.y - p.z*q.z - p.w*q.w;
	res.y = p.x*q.y + p.y*q.x + p.z*q.w - p.w*q.z;
	res.z = p.x*q.z + p.z*q.x + p.w*q.y - p.y*q.w;
	res.w = p.x*q.w + p.w*q.x + p.y*q.z - p.z*q.y;
	return res;
}

// Create quaternion for axis angle rotation
point quat_axis(point n, float theta) {
	theta /= 2.0f;
	float stheta = sin(theta);
	point res;
	res.x = n.x * stheta;
	res.y = n.y * stheta;
	res.z = n.z * stheta;
	res.w = cos(theta);
	return res;
}

// Conjugate quaternion
point quat_conj(point p) {
	point res;
	res.x = -p.x;
	res.y = -p.y;
	res.z = -p.z;
	res.w = p.w;
	return res;
}

// Normalize quaternion
point quat_norm(point p) {
	point res;
	float norm = 1.0f / sqrt(p.w*p.w+p.x*p.x+p.y*p.y+p.z*p.z);
	res.x = p.x * norm;
	res.y = p.y * norm;
	res.z = p.z * norm;
	res.w = p.w * norm;
	return res;
}

// Cross product
point quat_crossprod(point u, point v){
	point res;
	res.x = u.y*v.z-u.z*v.y;
	res.y = u.z*v.x-u.x*v.z;
	res.z = u.x*v.y-u.y*v.x;
	res.w = 0.0f;
	return res;
}

// Dot product
float quat_dotprod(point u, point v) {
	return u.x*v.x + u.y*v.y + u.z*v.z + u.w*v.w;
}

void dev_deflect(float* p, int np, float theta, float phi) {
	point d; d.x = p[0]; d.y = p[1]; d.z = p[2]; d.w = 0.0f;

	// compute the particle rotation
	point ref; ref.x = 0.0f; ref.y = 0.0f; ref.z = 1.0f; ref.w = 0.0f;
	point a = quat_norm(quat_crossprod(ref, d)); // get axis
	float alpha = -acos(quat_dotprod(ref, d));   // get angle
	point r = quat_axis(a, alpha);               // build axis-angle quaternion

	// compute quaternion for deflection
	phi += 4.712388980f; // 3*pi / 2.0
	a.x = cos(phi); a.y = sin(phi); a.z = 0.0f; a.w = 0.0f;
	point q = quat_axis(a, theta);

	// compose rotations
	point n = quat_mul(ref, quat_conj(q)); // n = q.ref.q*
	n = quat_mul(q, n);
	n = quat_mul(n, quat_conj(r));         // n = r.n.r*
	n = quat_mul(r, n);
	
	p[0] = n.x; p[1] = n.y; p[2] = n.z;
}

/*
// Apply deflection of the particle with quaternion stuff
void dev_deflect_(float* p, int np, float theta, float phi) {
	point d;
	d.x = p[0];	d.y = p[1];	d.z = p[2];	d.w = 0.0f;
	d = quat_norm(d);
	// select the best axis
	point a;
	a.x = 0.0f; a.y = 0.0f; a.z = 0.0f; a.w = 0.0f;
	if (d.x < d.y) {a.x = 1.0f;}
	else {a.y = 1.0f;}
	// create virtual axis given by p^a
	a = quat_crossprod(d, a);
	a = quat_norm(a);
	// build rotation around p axis with phi (in order to rotate the next rotation axis a)
	point r = quat_axis(d, phi);
	// do rotation of a = rar*
	a = quat_mul(a, quat_conj(r)); // a = ar*
	a = quat_mul(r, a);            // a = ra
	// build rotation around the axis with theta (thus rotate p)
	r = quat_axis(a, theta);
	// do final rotation of p = rpr*
	d = quat_mul(d, quat_conj(r));
	d = quat_mul(r, d);
	// assign new values
	p[0] = d.x; p[1] = d.y; p[2] = d.z;
}
*/
