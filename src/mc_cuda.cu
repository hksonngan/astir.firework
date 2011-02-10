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

#include "mc_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>
#include <cufft.h>
#include <sys/time.h>
#include <math_constants.h>

/***********************************************************
 * Utils
 ***********************************************************/
__constant__ const float pi = 3.14159265358979323846;
__constant__ const float twopi = 2*pi;

// Stack of gamma particles, format data is defined as SoA
struct StackGamma{
	float* E;
	float* dx;
	float* dy;
	float* dz;
	float* px;
	float* py;
	float* pz;
	int* seed;
	char* live;
	char* in;
	unsigned int size;
};

// Given by Hector doesn't work properly
__device__ float park_miller(long unsigned int *seed) {
	long unsigned int hi, lo;
	int const a = 16807;
	int const m = 2147483647;
	float const recm = __fdividef(1.0f, m);

	lo = a * (*seed & 0xFFFF);
	hi = a * (*seed >> 16);
	lo += (hi & 0x7FFF) << 16;
	lo += (hi >> 15);
	if (lo > 0x7FFFFFFF) {lo -= 0x7FFFFFFF;}
	*seed = (long)lo;

	return (*seed)*recm;
}

// Park-Miller from C numerical book
__device__ float park_miller_jb(int *seed) {
	int const a = 16807;
	int const m = 2147483647;
	int const iq = 127773;
	int const ir = 2836;
	int const mask = 123459876;
	float const recm = __fdividef(1.0f, m);
	int k;
	float ans;
	*seed ^= mask;
	k = (*seed)/iq;
	*seed = a * (*seed-k*iq) - ir*k;
	if (*seed < 0) {*seed += m;}
	ans = recm * (*seed);
	*seed ^= mask;
	return ans;
}

/***********************************************************
 * Physics
 ***********************************************************/
// kernel Compton Cross Section Per Atom
__device__ float Compton_CSPA(float E, float Z) {
	float CrossSection = 0.0;
	if (Z<0.9999f || E < 1e-4f) {return CrossSection;}

	float p1Z = Z*(2.7965e-23f + 1.9756e-27f*Z + -3.9178e-29f*Z*Z);
	float p2Z = Z*(-1.8300e-23f + -1.0205e-24f*Z + 6.8241e-27f*Z*Z);
	float p3Z = Z*(6.7527e-22f + -7.3913e-24f*Z + 6.0480e-27f*Z*Z);
	float p4Z = Z*(-1.9798e-21f + 2.7079e-24f*Z + 3.0274e-26f*Z*Z);
	float T0 = (Z < 1.5f)? 40.0e-3f : 15.0e-3f;
	float d1, d2, d3, d4, d5;

	d1 = __fdividef(fmaxf(E, T0), 0.510998910f); // X
	CrossSection = __fdividef(p1Z*__logf(1.0f+2.0f*d1), d1) + __fdividef(p2Z + p3Z*d1 + p4Z*d1*d1, 1.0f + 20.0f*d1 + 230.0f*d1*d1 + 440.0f*d1*d1*d1);

	if (E < T0) {
		d1 = __fdividef(T0+1.0e-3f, 0.510998910f); // X
		d2 = __fdividef(p1Z*__logf(1.0f+2.0f*d1), d1) + __fdividef(p2Z + p3Z*d1 + p4Z*d1*d1, 1.0f + 20.0f*d1 + 230.0f*d1*d1 + 440.0f*d1*d1*d1); // sigma
		d3 = __fdividef(-T0 * (d2 - CrossSection), CrossSection*1.0e-3f); // c1
		d4 = (Z > 1.5f)? 0.375f-0.0556f*__logf(Z) : 0.15f; // c2
		d5 = __logf(__fdividef(E, T0)); // y
		CrossSection *= __expf(-d5 * (d3 + d4*d5));
	}

	return CrossSection;
}

__device__ float Compton_mu_eau(float E) {
	return (2*Compton_CSPA(E, 1) + Compton_CSPA(E, 8)) * 3.342796664e+19f; // Avogadro*H2O_density / (2*a_H+a_O)
}

__device__ float Compton_mu_Al(float E) {
	return Compton_CSPA(E, 13) * 6.024030465e+19f; // Avogadro*Al_density/a_Al
}

/***********************************************************
 * Managment kernel
 ***********************************************************/
__global__ void kernel_particle_birth(StackGamma stackgamma, int3 dimvol) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x)+threadIdx.x;
	float phi, theta;
	int seed = stackgamma.seed[id];
	// warmpup to diverge
	park_miller_jb(&seed);
	park_miller_jb(&seed);
	park_miller_jb(&seed);
	if (id < stackgamma.size) {
		// position
		stackgamma.px[id] = park_miller_jb(&seed)*dimvol.x;
		stackgamma.py[id] = park_miller_jb(&seed)*dimvol.y;
		stackgamma.pz[id] = park_miller_jb(&seed)*dimvol.z;
		// direction
		phi = park_miller_jb(&seed) * twopi;
		theta = park_miller_jb(&seed) * pi - 0.5*pi;
		stackgamma.dx[id] = __cosf(theta) * __cosf(phi);
		stackgamma.dy[id] = __cosf(theta) * __sinf(phi);
		stackgamma.dz[id] = __sinf(theta);
		// enable particles
		stackgamma.live[id] = 1;
		stackgamma.in[id] = 1;
		stackgamma.seed[id] = seed;
	}
}

/***********************************************************
 * Tracking kernel
 ***********************************************************/
__global__ void kernel_siddon(float* dvol, int3 dimvol, StackGamma stackgamma) {

	int3 u, i, e, stepi, old;
	float3 p0, pe, stept, astart, run, delta;
	float pq, oldv, newv, totv, mu, val;
	float eps = 1.0e-5f;
	unsigned int id = __umul24(blockIdx.x, blockDim.x)+threadIdx.x;
	int jump = dimvol.x*dimvol.y;
	int seed = stackgamma.seed[id];
	int inside;
	
	if (id < stackgamma.size) {
		p0.x = stackgamma.px[id];
		p0.y = stackgamma.py[id];
		p0.z = stackgamma.pz[id];
		delta.x = stackgamma.dx[id];
		delta.y = stackgamma.dy[id];
		delta.z = stackgamma.dz[id];
		
		pq = -__logf(park_miller_jb(&seed)) / 0.018f;
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

		mu = oldv*dvol[e.z*jump + e.y*dimvol.x + e.x];
		dvol[e.z*jump + e.y*dimvol.x + e.x] += oldv;
		
		totv = 0.0f;
		old.x = i.x;
		old.y = i.y;
		old.z = i.z;
		inside = 1;
		while ((oldv < pq) & inside) {
			newv = run.x;
			if (run.y < newv) {newv=run.y;}
			if (run.z < newv) {newv=run.z;}
			val = (newv - oldv);
			mu = val * dvol[i.z*jump + i.y*dimvol.x + i.x];
			dvol[i.z*jump + i.y*dimvol.x + i.x] += val;
			totv += val;
			oldv = newv;
			old.x = i.x;
			old.y = i.y;
			old.z = i.z;
			if (run.x==newv) {i.x += stepi.x; run.x += stept.x;}
			if (run.y==newv) {i.y += stepi.y; run.y += stept.y;}
			if (run.z==newv) {i.z += stepi.z; run.z += stept.z;}
			inside = (i.x >= 0) & (i.x < dimvol.x) & (i.y >= 0) & (i.y < dimvol.y) & (i.z >= 0) & (i.z < dimvol.z);
		}

		if (!inside) {
			stackgamma.in[id] = 0;
			return;
		}
		
		mu = (pq-totv)*dvol[old.z*jump + old.y*dimvol.x + old.x];
		dvol[old.z*jump + old.y*dimvol.x + old.x] += (pq - totv);

		stackgamma.seed[id] = seed;

	} // id < nx

}

/*
__global__ void kernel_amanatides(float* dvol, float* dX0, float* dY0, float* dZ0,
								  float* dXe, float* dYe, float* dZe, int nx0, int jump, int nx) {

	int3 u, i, e, stepi;
	float3 p0, pe, stept, astart, run;
	float pq, oldv, totv, mu, val;
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

		mu = oldv*dvol[e.z*jump + e.y*nx + e.x];
		//dvol[e.z*jump + e.y*nx + e.x] += oldv;
		
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
			mu = val * dvol[i.z*jump + i.y*nx + i.x];
			//dvol[i.z*jump + i.y*nx + i.x] += val;
			oldv = totv;
		}
		
		mu = (pq-totv)*dvol[i.z*jump + i.y*nx + i.x];
		//dvol[i.z*jump + i.y*nx + i.x] += (pq - totv);

	} // id < nx

}
*/
/*
__global__ void kernel_raypro(float* dvol, int3 dimvol, StackGamma stackgamma) {
	float3 xi, x0, d, rd, db, sd;
	int3 p, b, ob;
	float t, tn, tot_t, dist, mu, phi, theta;
	float eps = 1.0e-5f;
	int jump = dimvol.x*dimvol.y;

	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int seed = stackgamma.seed[id];
	int inside;
	int watchdog;
	if (id < stackgamma.size) {
		x0.x = stackgamma.px[id];
		x0.y = stackgamma.py[id];
		x0.z = stackgamma.pz[id];
		d.x = stackgamma.dx[id];
		d.y = stackgamma.dy[id];
		d.z = stackgamma.dz[id];

		dist = -__logf(park_miller_jb(&seed)) / 0.018f;

		if (d.x==0) {d.x=eps;}
		if (d.y==0) {d.y=eps;}
		if (d.z==0) {d.z=eps;}

		rd.x = __fdividef(1.0f, d.x);
		rd.y = __fdividef(1.0f, d.y);
		rd.z = __fdividef(1.0f, d.z);

		db.x = (d.x > 0) - (d.x < 0) * eps;
		db.y = (d.y > 0) - (d.y < 0) * eps;
		db.z = (d.z > 0) - (d.z < 0) * eps;

		b.x = int(x0.x+db.x);
		b.y = int(x0.y+db.y);
		b.z = int(x0.z+db.z);
		ob.x = b.x; ob.y = b.y; ob.z = b.z;

		t = (b.x - x0.x) * rd.x;
		tn = (b.y - x0.y) * rd.y;
		t = fminf(t, tn);
		tn = (b.z - x0.z) * rd.z;
		t = fminf(t, tn);

		xi.x = x0.x + (d.x * t);
		xi.y = x0.y + (d.y * t);
		xi.z = x0.z + (d.z * t);

		tn = 1.0f + int(xi.x) - xi.x;
		xi.x += (tn * (tn < eps));
		tn = 1.0f + int(xi.y) - xi.y;
		xi.y += (tn * (tn < eps));
		tn = 1.0f + int(xi.z) - xi.z;
		xi.z += (tn * (tn < eps));

		tot_t = t;
		p.x = int(x0.x);
		p.y = int(x0.y);
		p.z = int(x0.z);

		inside = 1;
		watchdog=0;
		while ((tot_t < dist) & inside) {
			mu = t * dvol[p.z*jump + p.y*dimvol.x + p.x];
			//dvol[p.z*jump + p.y*dimvol.x + p.x] += t;
			
			b.x = int(xi.x + db.x);
			b.y = int(xi.y + db.y);
			b.z = int(xi.z + db.z);

			t = (b.x - xi.x) * rd.x;
			tn = (b.y - xi.y) * rd.y;
			t = fminf(t, tn);
			tn = (b.z - xi.z) * rd.z;
			t = fminf(t, tn);
			
			xi.x = xi.x + (d.x * t);
			xi.y = xi.y + (d.y * t);
			xi.z = xi.z + (d.z * t);

			tot_t += t;
			p.x += (b.x - ob.x);
			p.y += (b.y - ob.y);
			p.z += (b.z - ob.z);
			ob.x = b.x; ob.y = b.y; ob.z = b.z;
			
			inside = (p.x >= 0) & (p.x < dimvol.x) & (p.y >= 0) & (p.y < dimvol.y) & (p.z >= 0) & (p.z < dimvol.z);
			dvol[watchdog] = p.x;
			++watchdog;
			if (watchdog > 500) {
				//dvol[0] = b.x;
				//dvol[1] = p.z;
				break;
			}

		}

		if (!inside) {
			stackgamma.in[id] = 0;
			return;
		}

		mu = (dist-tot_t) * dvol[p.z*jump + p.y*dimvol.x + p.x];
		//dvol[p.z*jump + p.y*dimvol.x + p.x] += (dist-tot_t);

		stackgamma.seed[id] = seed;
		
	} // id
	
}
*/

/***********************************************************
 * Main
 ***********************************************************/
void mc_cuda(float* vol, int nz, int ny, int nx, int nparticles) {
	cudaSetDevice(1);

    timeval start, end;
    double t1, t2, diff;
	int3 dimvol;
	int n;
	
	dimvol.x = nx;
	dimvol.y = ny;
	dimvol.z = nz;

	// Volume allocation
	unsigned int mem_vol = nz*ny*nx * sizeof(float);
	float* dvol;
	cudaMalloc((void**) &dvol, mem_vol);
	cudaMemset(dvol, 0, mem_vol);

	// Stack allocation memory
	StackGamma stackgamma;
	stackgamma.size = nparticles;
	unsigned int mem_stack_float = stackgamma.size * sizeof(float);
	unsigned int mem_stack_int = stackgamma.size * sizeof(int);
	unsigned int mem_stack_char = stackgamma.size * sizeof(char);
	cudaMalloc((void**) &stackgamma.E, mem_stack_float);
	cudaMalloc((void**) &stackgamma.dx, mem_stack_float);
	cudaMalloc((void**) &stackgamma.dy, mem_stack_float);
	cudaMalloc((void**) &stackgamma.dz, mem_stack_float);
	cudaMalloc((void**) &stackgamma.px, mem_stack_float);
	cudaMalloc((void**) &stackgamma.py, mem_stack_float);
	cudaMalloc((void**) &stackgamma.pz, mem_stack_float);
	cudaMalloc((void**) &stackgamma.seed, mem_stack_int);
	cudaMalloc((void**) &stackgamma.live, mem_stack_char);
	cudaMalloc((void**) &stackgamma.in, mem_stack_char);

	// Init seeds
	int* tmp = (int*)malloc(stackgamma.size * sizeof(int));
	srand(10);
	n=0;
	while (n<stackgamma.size) {tmp[n] = rand(); ++n;}
	cudaMemcpy(stackgamma.seed, tmp, mem_stack_int, cudaMemcpyHostToDevice);
	free(tmp);

	// Vars kernel
	dim3 threads, grid;
	int block_size = 256;
	int grid_size = (nparticles + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	
	// Init particles
    gettimeofday(&start, NULL);
    t1 = start.tv_sec + start.tv_usec / 1000000.0;
	kernel_particle_birth<<<grid, threads>>>(stackgamma, dimvol);
	cudaThreadSynchronize();
    gettimeofday(&end, NULL);
    t2 = end.tv_sec + end.tv_usec / 1000000.0;
    diff = t2 - t1;
    printf("Create gamma particles %f s\n", diff);
	
	// Propagation
	gettimeofday(&start, NULL);
    t1 = start.tv_sec + start.tv_usec / 1000000.0;
	kernel_siddon<<<grid, threads>>>(dvol, dimvol, stackgamma);
	cudaThreadSynchronize();
    gettimeofday(&end, NULL);
    t2 = end.tv_sec + end.tv_usec / 1000000.0;
    diff = t2 - t1;
    printf("Track gamma particles %f s\n", diff);
	
	//cudaMemcpy(tmp, stackgamma.seed, mem_stack_int, cudaMemcpyDeviceToHost);
	cudaMemcpy(vol, dvol, mem_vol, cudaMemcpyDeviceToHost);
	//printf("new seed %i\n", tmp[0]);

	cudaFree(dvol);
	cudaFree(stackgamma.E);
	cudaFree(stackgamma.dx);
	cudaFree(stackgamma.dy);
	cudaFree(stackgamma.dz);
	cudaFree(stackgamma.px);
	cudaFree(stackgamma.py);
	cudaFree(stackgamma.pz);
	cudaFree(stackgamma.live);
	cudaFree(stackgamma.in);
	cudaFree(stackgamma.seed);	
	cudaThreadExit();

}
