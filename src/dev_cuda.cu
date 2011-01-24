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

#define INF 2e10f;
#define rnd(x) (x*rand() / RAND_MAX)

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
