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

#include "dev_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>
#include <cufft.h>
#include <sys/time.h>
#include <math_constants.h>

// here, put your code!

/***********************************************
 * Utils
 ***********************************************/
#define INF 2e10f;
#define rnd(x) (x*rand() / RAND_MAX)

/***********************************************
 * Test ray-tracing
 ***********************************************/
struct Sphere{
	float lum;
	float radius;
	float x, y, z;
	__device__ float hit(float ox, float oy, float *n) {
		float dx = ox - x;
		float dy = oy - y;
		if (dx*dx + dy*dy < radius*radius) {
			float dz = sqrtf(radius*radius - dx*dx - dy*dy);
			*n = dz / sqrtf(radius*radius);
			return dz + z;
		}
		return -INF;
	}
};

__global__ void kernel_sphere_ray(Sphere *s, float *dim, int ns, int ny, int nx) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int y = id / nx;
	int x = id % nx;
	float ox = (x - nx / 2);
	float oy = (y - ny / 2);
	float lum = 0.0f;
	float maxz = -INF;
	for (int i=0; i<ns; ++i) {
		float n;
		float t=s[i].hit(ox, oy, &n);
		if (t>maxz) {
			float fscale = n;
			lum = s[i].lum * fscale;
		}
	}
	dim[y*nx + x] = lum;
}

void dev_raytracing(float* im, int nim1, int nim2, int ns) {
	int npix = nim1*nim2;
	Sphere *s;
	cudaMalloc((void**) &s, sizeof(Sphere) * ns);
	float *dim;
	cudaMalloc((void**) &dim, sizeof(float) * npix);
	Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * ns);
	for (int i=0; i<ns; ++i) {
		temp_s[i].lum = rnd(1.0f);
		temp_s[i].x = rnd(300.0f) - 150.0f;
		temp_s[i].y = rnd(300.0f) - 150.0f;
		temp_s[i].z = rnd(300.0f) - 150.0f;
		temp_s[i].radius = rnd(50.0f) + 1.0f;
	}
	cudaMemcpy(s, temp_s, sizeof(Sphere) * ns, cudaMemcpyHostToDevice);
	cudaMemcpy(dim, im, sizeof(float) * npix, cudaMemcpyHostToDevice);
	free(temp_s);
	dim3 threads, grid;
	int block_size = 256;
	int grid_size = (npix + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	kernel_sphere_ray<<<grid, threads>>>(s, dim, ns, nim1, nim2);
	cudaMemcpy(im, dim, sizeof(float) * npix, cudaMemcpyDeviceToHost);
	cudaFree(dim);
	cudaFree(s);
}

/***********************************************
 * Test divergence
 ***********************************************/

__global__ void kernel_div_v1(float* dA, float* dB, float* dC, float* dres, int na) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int va, vb, vc, vres;
	if (id < na) {
		va = dA[id];
		vb = dB[id];
		vc = dC[id];
		
		vres = va;
		if (vb > vres) {vres = vb;}
		if (vc > vres) {vres = vc;}
		
		dres[id] = vres;		
	}
}

__global__ void kernel_div_v2(float* dA, float* dB, float* dC, float* dres, int na) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int va, vb, vc;
	int fa;
	int vres;
	if (id < na) {
		va = dA[id];
		vb = dB[id];
		vc = dC[id];
		
		fa = va>vb;
		vres = fa*va + !fa*vb;
		fa = vres>vc;
		vres = vres*fa + !fa*vc;
		
		dres[id] = vres;
	}

}

void dev_div(float* A, int na, float* B, int nb, float* C, int nc, float* res, int nres) {
	cudaSetDevice(0);
	unsigned int mem_vec = na * sizeof(float);
	float *dA;
	float *dB;
	float *dC;
	float *dres;
	cudaMalloc((void**) &dA, mem_vec);
	cudaMalloc((void**) &dB, mem_vec);
	cudaMalloc((void**) &dC, mem_vec);
	cudaMalloc((void**) &dres, mem_vec);
	cudaMemcpy(dA, A, mem_vec, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, mem_vec, cudaMemcpyHostToDevice);
	cudaMemcpy(dC, C, mem_vec, cudaMemcpyHostToDevice);
	cudaMemcpy(dres, res, mem_vec, cudaMemcpyHostToDevice);
	dim3 threads, grid;
	int block_size = 256;
	int grid_size = (na + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	for (int i=0; i<100000; ++i) {
		kernel_div_v2<<<grid, threads>>>(dA, dB, dC, dres, na);
	}
	cudaMemcpy(res, dres, mem_vec, cudaMemcpyDeviceToHost);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	cudaFree(dres);
	cudaThreadExit();
}

/***********************************************
 * Test ray-projector
 ***********************************************/

__global__ void kernel_amanatides(float* dvol, float* dX0, float* dY0, float* dZ0,
								  float* dXe, float* dYe, float* dZe, int nx0, int jump, int nx) {

	int3 u, i, e, stepi;
	float3 p0, pe, stept, astart, run;
	float pq, oldv, totv, val;
	float eps = 1.0e-5f;
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < nx0) {
		p0.x = dX0[id];
		p0.y = dY0[id];
		p0.z = dZ0[id];
		pe.x = dXe[id];
		pe.y = dYe[id];
		pe.z = dZe[id];

		e.x = int(p0.x);
		e.y = int(p0.y);
		e.z = int(p0.z);

		if ((pe.x-p0.x) > 0) {stepi.x = 1; u.x = e.x + 1;}
		if ((pe.x-p0.x) < 0) {stepi.x = -1; u.x = e.x;}
		if ((pe.x-p0.x) == 0) {stepi.x = 0; u.x = e.x; pe.x = eps;}

		if ((pe.y-p0.y) > 0) {stepi.y = 1; u.y = e.y+1;}
		if ((pe.y-p0.y) < 0) {stepi.y = -1; u.y = e.y;}
		if ((pe.y-p0.y) == 0) {stepi.y = 0; u.y = e.y; pe.y = eps;}

		if ((pe.z-p0.z) > 0) {stepi.z = 1; u.z = e.z+1;}
		if ((pe.z-p0.z) < 0) {stepi.z = -1; u.z = e.z;}
		if ((pe.z-p0.z) == 0) {stepi.z = 0; u.z = e.z; pe.z = eps;}

		astart.x = (u.x - p0.x) / (pe.x - p0.x);
		astart.y = (u.y - p0.y) / (pe.y - p0.y);
		astart.z = (u.z - p0.z) / (pe.z - p0.z);
		
		pq = sqrtf((p0.x-pe.x)*(p0.x-pe.x)+(p0.y-pe.y)*(p0.y-pe.y)+(p0.z-pe.z)*(p0.z-pe.z));
		run.x = astart.x * pq;
		run.y = astart.y * pq;
		run.z = astart.z * pq;
		oldv = run.x;
		if (run.y < oldv) {oldv = run.y;}
		if (run.z < oldv) {oldv = run.z;}

		stept.x = fabsf((pq / (pe.x - p0.x)));
		stept.y = fabsf((pq / (pe.y - p0.y)));
		stept.z = fabsf((pq / (pe.z - p0.z)));
		i.x = e.x;
		i.y = e.y;
		i.z = e.z;

		//mu = oldv*dvol[e.z*jump + e.y*nx + e.x];
		dvol[e.z*jump + e.y*nx + e.x] += oldv;
		
		totv = 0.0f;
		while (totv < pq) {
			if (run.x < run.y) {
				if (run.x < run.z) {i.x += stepi.x; run.x += stept.x;}
				else {i.z += stepi.z; run.z += stept.z;}
			} else {
				if (run.y < run.z) {i.y += stepi.y; run.y += stept.y;}
				else {i.z += stepi.z; run.z += stept.z;}
			}
			totv = run.x;
			if (run.y < totv) {totv=run.y;}
			if (run.z < totv) {totv=run.z;}
			val = totv-oldv;
			//mu = val * dvol[i.z*jump + i.y*nx + i.x];
			dvol[i.z*jump + i.y*nx + i.x] += val;
			oldv = totv;
		}
		
		//mu = (pq-totv)*dvol[i.z*jump + i.y*nx + i.x];
		dvol[i.z*jump + i.y*nx + i.x] += (pq - totv);

	} // id < nx

}

/*
__global__ void kernel_siddon(int3 dimvol, StackGamma stackgamma, float* dtrack, float dimvox) {

	int3 u, i, e, stepi;
	float3 p0, pe, stept, astart, run, delta;
	float pq, oldv, newv, totv, val, E;
	float eps = 1.0e-5f;
	unsigned int id = __umul24(blockIdx.x, blockDim.x)+threadIdx.x;
	int jump = dimvol.x*dimvol.y;
	int seed, inside, oldmat, mat;

	// debug
	//int j=0;
	
	if (id < stackgamma.size) {
		p0.x = stackgamma.px[id];
		p0.y = stackgamma.py[id];
		p0.z = stackgamma.pz[id];
		delta.x = stackgamma.dx[id];
		delta.y = stackgamma.dy[id];
		delta.z = stackgamma.dz[id];
		seed = stackgamma.seed[id];
		E = stackgamma.E[id];

		// get free mean path
		//oldmat = dvol[int(p0.z)*jump + int(p0.y)*dimvol.x + int(p0.x)];
		oldmat = tex1Dfetch(tex_vol, int(p0.z)*jump + int(p0.y)*dimvol.x + int(p0.x));
		pq = -__fdividef(__logf(park_miller_jb(&seed)), att_from_mat(oldmat, E));
		pq = __fdividef(pq, dimvox);

		//dtrack[id]=pq;
		//++j;
		
		pe.x = p0.x + delta.x*pq;
		pe.y = p0.y + delta.y*pq;
		pe.z = p0.z + delta.z*pq;

		e.x = int(p0.x);
		e.y = int(p0.y);
		e.z = int(p0.z);

		if ((pe.x-p0.x) > 0) {stepi.x = 1; u.x = e.x + 1;}
		if ((pe.x-p0.x) < 0) {stepi.x = -1; u.x = e.x;}
		if ((pe.x-p0.x) == 0) {stepi.x = 0; u.x = e.x; pe.x = eps;}

		if ((pe.y-p0.y) > 0) {stepi.y = 1; u.y = e.y+1;}
		if ((pe.y-p0.y) < 0) {stepi.y = -1; u.y = e.y;}
		if ((pe.y-p0.y) == 0) {stepi.y = 0; u.y = e.y; pe.y = eps;}

		if ((pe.z-p0.z) > 0) {stepi.z = 1; u.z = e.z+1;}
		if ((pe.z-p0.z) < 0) {stepi.z = -1; u.z = e.z;}
		if ((pe.z-p0.z) == 0) {stepi.z = 0; u.z = e.z; pe.z = eps;}

		astart.x = (u.x - p0.x) / (pe.x - p0.x);
		astart.y = (u.y - p0.y) / (pe.y - p0.y);
		astart.z = (u.z - p0.z) / (pe.z - p0.z);

		run.x = astart.x * pq;
		run.y = astart.y * pq;
		run.z = astart.z * pq;
		oldv = run.x;
		if (run.y < oldv) {oldv = run.y;}
		if (run.z < oldv) {oldv = run.z;}

		stept.x = fabsf((pq / (pe.x - p0.x)));
		stept.y = fabsf((pq / (pe.y - p0.y)));
		stept.z = fabsf((pq / (pe.z - p0.z)));
		i.x = e.x;
		i.y = e.y;
		i.z = e.z;

		if (run.x == oldv) {run.x += stept.x; i.x += stepi.x;}
		if (run.y == oldv) {run.y += stept.y; i.y += stepi.y;}
		if (run.z == oldv) {run.z += stept.z; i.z += stepi.z;}

		// to debug
		//dtrack[e.z*jump + e.y*dimvol.x + e.x] += oldv;
		
		totv = 0.0f;
		inside = 1;
		while ((oldv < pq) & inside) {
			newv = run.x;
			if (run.y < newv) {newv=run.y;}
			if (run.z < newv) {newv=run.z;}
			val = (newv - oldv);
			
			// if mat change
			//mat = dvol[i.z*jump + i.y*dimvol.x + i.x];
			mat = tex1Dfetch(tex_vol, i.z*jump + i.y*dimvol.x + i.x);
			if (mat != oldmat) {
				pq = oldv;
				pq += -__fdividef(__logf(park_miller_jb(&seed)), att_from_mat(mat, E));
				oldmat = mat;
				//dtrack[i.z*jump + i.y*dimvol.x + i.x] += 2.0f;
				//++j;
				//dtrack[id]=1.0f/att_from_mat(mat, E);
			}

			totv += val;
			oldv = newv;
			if (run.x==newv) {i.x += stepi.x; run.x += stept.x;}
			if (run.y==newv) {i.y += stepi.y; run.y += stept.y;}
			if (run.z==newv) {i.z += stepi.z; run.z += stept.z;}
			inside = (i.x >= 0) & (i.x < dimvol.x) & (i.y >= 0) & (i.y < dimvol.y) & (i.z >= 0) & (i.z < dimvol.z);
			// debug
			dtrack[i.z*jump + i.y*dimvol.x + i.x] += 1.0f;
		}

		pe.x = p0.x + delta.x*oldv;
		pe.y = p0.y + delta.y*oldv;
		pe.z = p0.z + delta.z*oldv;
		stackgamma.seed[id] = seed;
		stackgamma.px[id] = pe.x;
		stackgamma.py[id] = pe.y;
		stackgamma.pz[id] = pe.z;

		// to debug
		//dtrack[int(pe.z)*jump + int(pe.y)*dimvol.x + int(pe.x)] += 1.0f;

		if (!inside) {stackgamma.in[id] = 0;}

	} // id < nx

}
*/
