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

#include "pet_cuda.h"
#include <stdio.h>
#include <cublas.h>
#include <cufft.h>
#include <sys/time.h>
#include <math_constants.h>

// textures
texture<float, 1, cudaReadModeElementType> tex_im;
texture<float, 1, cudaReadModeElementType> tex_at;
texture<unsigned short, 1, cudaReadModeElementType> tex_x1;
texture<unsigned short, 1, cudaReadModeElementType> tex_y1;
texture<unsigned short, 1, cudaReadModeElementType> tex_z1;
texture<unsigned short, 1, cudaReadModeElementType> tex_x2;
texture<unsigned short, 1, cudaReadModeElementType> tex_y2;
texture<unsigned short, 1, cudaReadModeElementType> tex_z2;

__device__ inline void atomicFloatAdd(float* address, float val) {
	int i_val = __float_as_int(val);
	int tmp0 = 0;
	int tmp1;

	while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0)
	{
		tmp0 = tmp1;
		i_val = __float_as_int(val + __int_as_float(tmp1));
	}
}

/*********************************************
 *  PET 3D LM-EM
 *********************************************/

// kernel to raytrace 3D line in SRM with DDA algorithm and compute F on-line
__global__ void pet3D_SRM_DDA_F_ON(unsigned int* d_F, int wim, int nx1, int nim, float scale) {

	int length, n, diffx, diffy, diffz, step;
	float flength, x, y, z, lx, ly, lz, xinc, yinc, zinc, Qi;
	unsigned short int x1, y1, z1;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	step = wim*wim;
	
	if (idx < nx1) {
		Qi = 0.0f;
		x1 = tex1Dfetch(tex_x1, idx);
		y1 = tex1Dfetch(tex_y1, idx);
		z1 = tex1Dfetch(tex_z1, idx);
		diffx = tex1Dfetch(tex_x2, idx)-x1;
		diffy = tex1Dfetch(tex_y2, idx)-y1;
		diffz = tex1Dfetch(tex_z2, idx)-z1;
		lx = abs(diffx);
		ly = abs(diffy);
		lz = abs(diffz);
		length = ly;
		if (lx > length) {length = lx;}
		if (lz > length) {length = lz;}
		flength = 1.0f / (float)length;
		xinc = diffx * flength;
		yinc = diffy * flength;
		zinc = diffz * flength;
		x = x1;
		y = y1;
		z = z1;
		for (n=0; n<=length; ++n) {
			Qi = Qi + tex1Dfetch(tex_im, (int)z * step + (int)y * wim + (int)x);
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}

		// compute F
		if (Qi==0.0f) {return;}
		Qi = 1 / Qi;
		x = x1;
		y = y1;
		z = z1;
		for (n=0; n<=length; ++n) {
			//atomicFloatAdd(&d_F[(int)z * step + (int)y * wim + (int)x], Qi);
			atomicAdd(&d_F[(int)z * step + (int)y * wim + (int)x], (unsigned int)(Qi*scale));
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
	}
}

// Compute update in LM 3D-OSEM algorithm on-line with DDA line drawing
void kernel_pet3D_LMOSEM_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
							  unsigned short int* z1, int nz1,	unsigned short int* x2, int nx2,
							  unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
							  float* im, int nim1, int nim2, int nim3, float* F, int nf1, int nf2, int nf3,
							  int wim, int ID){

	// select a GPU
	if (ID != -1){cudaSetDevice(ID);}
	// vars
	int block_size, grid_size, i;
	dim3 threads, grid;
	int nim = nim1 * nim2 * nim3;
	// Need to change
	int* Fi = (int*)calloc(nim, sizeof(int));
	// allocate device memory
	unsigned int mem_size_im = nim * sizeof(float);
	unsigned int mem_size_F = nim * sizeof(unsigned int);
	unsigned int mem_size_point = nx1 * sizeof(unsigned short int);
	float* d_im;
	unsigned int* d_F;
	unsigned short int* d_x1;
	unsigned short int* d_x2;
	unsigned short int* d_y1;
	unsigned short int* d_y2;
	unsigned short int* d_z1;
	unsigned short int* d_z2;
	cudaMalloc((void**) &d_im, mem_size_im);
	cudaMalloc((void**) &d_F, mem_size_F);
	cudaMalloc((void**) &d_x1, mem_size_point);
	cudaMalloc((void**) &d_y1, mem_size_point);
	cudaMalloc((void**) &d_z1, mem_size_point);
	cudaMalloc((void**) &d_x2, mem_size_point);
	cudaMalloc((void**) &d_y2, mem_size_point);
	cudaMalloc((void**) &d_z2, mem_size_point);
	// copy from host to device
	cudaMemcpy(d_im, im, mem_size_im, cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, Fi, mem_size_F, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x1, x1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y1, y1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z1, z1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2, x2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y2, y2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z2, z2, mem_size_point, cudaMemcpyHostToDevice);
	// prepare texture
	cudaBindTexture(NULL, tex_im, d_im, mem_size_im);
	cudaBindTexture(NULL, tex_x1, d_x1, mem_size_point);
	cudaBindTexture(NULL, tex_y1, d_y1, mem_size_point);
	cudaBindTexture(NULL, tex_z1, d_z1, mem_size_point);
	cudaBindTexture(NULL, tex_x2, d_x2, mem_size_point);
	cudaBindTexture(NULL, tex_y2, d_y2, mem_size_point);
	cudaBindTexture(NULL, tex_z2, d_z2, mem_size_point);
	// float to int scale
	float scale = 4000.0f;
	// kernel
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size; // CODE IS LIMITED TO < 16e6 lines
	threads.x = block_size;
	grid.x = grid_size;
	pet3D_SRM_DDA_F_ON<<<grid, threads>>>(d_F, wim, nx1, nim, scale);
	// get back F and convert
	cudaMemcpy(Fi, d_F, nim*sizeof(float), cudaMemcpyDeviceToHost);
	scale = 1 / scale;
	for (i=0; i<nim; ++i) {F[i] = (float)Fi[i] * scale;}
	// Free mem
	free(Fi);
	cudaFree(d_im);
	cudaFree(d_F);
	cudaFree(d_x1);
	cudaFree(d_y1);
	cudaFree(d_z1);
	cudaFree(d_x2);
	cudaFree(d_y2);
	cudaFree(d_z2);
	cudaThreadExit();
}

// Same as pet3D_SRM_DDA_F_ON with attenuation correction
__global__ void pet3D_SRM_DDA_F_ATT_ON(unsigned int* d_F, int wim, int nx1, int nim, float scale) {

	int length, n, diffx, diffy, diffz, step, ind;
	float flength, x, y, z, lx, ly, lz, xinc, yinc, zinc, Qi, Ai;
	unsigned short int x1, y1, z1;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	step = wim*wim;
	
	if (idx < nx1) {
		Qi = 0.0f;
		Ai = 0.0f;
		x1 = tex1Dfetch(tex_x1, idx);
		y1 = tex1Dfetch(tex_y1, idx);
		z1 = tex1Dfetch(tex_z1, idx);
		diffx = tex1Dfetch(tex_x2, idx)-x1;
		diffy = tex1Dfetch(tex_y2, idx)-y1;
		diffz = tex1Dfetch(tex_z2, idx)-z1;
		lx = abs(diffx);
		ly = abs(diffy);
		lz = abs(diffz);
		length = ly;
		if (lx > length) {length = lx;}
		if (lz > length) {length = lz;}
		flength = 1.0f / (float)length;
		xinc = diffx * flength;
		yinc = diffy * flength;
		zinc = diffz * flength;
		x = x1 + 0.5f;
		y = y1 + 0.5f;
		z = z1 + 0.5f;
		for (n=0; n<=length; ++n) {
			ind = (int)z * step + (int)y * wim + (int)x;
			Qi = Qi + tex1Dfetch(tex_im, ind);
			Ai = Ai - tex1Dfetch(tex_at, ind);
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
		// compute F
		if (Qi==0.0f) {return;}
		if (Ai < -5.0f) {Ai = -5.0f;}
		Qi = Qi * __expf(Ai);
		Qi = 1 / Qi;
		x = x1 + 0.5f;
		y = y1 + 0.5f;
		z = z1 + 0.5f;
		for (n=0; n<=length; ++n) {
			//atomicFloatAdd(&d_F[(int)z * step + (int)y * wim + (int)x], Qi);
			atomicAdd(&d_F[(int)z * step + (int)y * wim + (int)x], (unsigned int)(Qi*scale));
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
	}
}

// DEV Compute update in LM 3D-OSEM algorithm on-line with DDA line drawing and attenuation
void kernel_pet3D_LMOSEM_att_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
								  unsigned short int* z1, int nz1,	unsigned short int* x2, int nx2,
								  unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
								  float* im, int nim1, int nim2, int nim3,
								  float* F, int nf1, int nf2, int nf3,
								  float* mumap, int nmu1, int nmu2, int nmu3, int wim, int ID){

	// select a GPU
	if (ID != -1){cudaSetDevice(ID);}
	// vars
	int block_size, grid_size, i;
	dim3 threads, grid;
	int nim = nim1 * nim2 * nim3;
	// Need to change
	int* Fi = (int*)calloc(nim, sizeof(int));
	// allocate device memory
	unsigned int mem_size_im = nim * sizeof(float);
	unsigned int mem_size_F = nim * sizeof(unsigned int);
	unsigned int mem_size_point = nx1 * sizeof(unsigned short int);
	float* d_im;
	float* d_mumap;
	unsigned int* d_F;
	unsigned short int* d_x1;
	unsigned short int* d_x2;
	unsigned short int* d_y1;
	unsigned short int* d_y2;
	unsigned short int* d_z1;
	unsigned short int* d_z2;
	cudaMalloc((void**) &d_im, mem_size_im);
	cudaMalloc((void**) &d_mumap, mem_size_im);
	cudaMalloc((void**) &d_F, mem_size_F);
	cudaMalloc((void**) &d_x1, mem_size_point);
	cudaMalloc((void**) &d_y1, mem_size_point);
	cudaMalloc((void**) &d_z1, mem_size_point);
	cudaMalloc((void**) &d_x2, mem_size_point);
	cudaMalloc((void**) &d_y2, mem_size_point);
	cudaMalloc((void**) &d_z2, mem_size_point);
	// copy from host to device
	cudaMemcpy(d_im, im, mem_size_im, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mumap, mumap, mem_size_im, cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, Fi, mem_size_F, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x1, x1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y1, y1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z1, z1, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2, x2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y2, y2, mem_size_point, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z2, z2, mem_size_point, cudaMemcpyHostToDevice);
	// prepare texture
	cudaBindTexture(NULL, tex_im, d_im, mem_size_im);
	cudaBindTexture(NULL, tex_at, d_mumap, mem_size_im);
	cudaBindTexture(NULL, tex_x1, d_x1, mem_size_point);
	cudaBindTexture(NULL, tex_y1, d_y1, mem_size_point);
	cudaBindTexture(NULL, tex_z1, d_z1, mem_size_point);
	cudaBindTexture(NULL, tex_x2, d_x2, mem_size_point);
	cudaBindTexture(NULL, tex_y2, d_y2, mem_size_point);
	cudaBindTexture(NULL, tex_z2, d_z2, mem_size_point);
	// float to int scale
	float scale = 4000.0f;
	// kernel
	block_size = 256;
	grid_size = (nx1 + block_size - 1) / block_size; // CODE IS LIMITED TO < 16e6 lines
	threads.x = block_size;
	grid.x = grid_size;
	//pet3D_SRM_DDA_F_ON<<<grid, threads>>>(d_F, wim, nx1, nim, scale);
	pet3D_SRM_DDA_F_ATT_ON<<<grid, threads>>>(d_F, wim, nx1, nim, scale);
	// get back F and convert
	cudaMemcpy(Fi, d_F, nim*sizeof(float), cudaMemcpyDeviceToHost);
	for (i=0; i<nim; ++i) {F[i] = (float)Fi[i] / scale;}
	// Free mem
	free(Fi);
	cudaFree(d_im);
	cudaFree(d_mumap);
	cudaFree(d_F);
	cudaFree(d_x1);
	cudaFree(d_y1);
	cudaFree(d_z1);
	cudaFree(d_x2);
	cudaFree(d_y2);
	cudaFree(d_z2);
	cudaThreadExit();
}

/***********************************************
 * PET 3D OPLEM
 ***********************************************/

// Update the volume
__global__ void	pet3D_OPLEM_update(float* d_im, unsigned int* d_F, float* d_NM, float invscale, int nim) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nim) {
		d_im[idx] = d_im[idx] * (float)d_F[idx] * invscale * d_NM[idx];
		//if (d_im[idx] > 100.0f) {d_im[idx] = 100.0f;}
		d_F[idx] = 0;
	}
}

// DDA ray-projector
__global__ void pet3D_OPLEM_DDA(unsigned int* d_F, int sublor_start, int sublor_stop, int nim3,
								int step, int nim, int nsublor, float scale) {

	int length, n, diffx, diffy, diffz;
	float flength, x, y, z, lx, ly, lz, xinc, yinc, zinc, Qf;
	unsigned short int x1, y1, z1;
	unsigned int Qi;
	int idx = blockIdx.x * blockDim.x + threadIdx.x + sublor_start;
	
	if (idx < sublor_stop) {
		Qf = 0.0f;
		x1 = tex1Dfetch(tex_x1, idx);
		y1 = tex1Dfetch(tex_y1, idx);
		z1 = tex1Dfetch(tex_z1, idx);
		diffx = tex1Dfetch(tex_x2, idx)-x1;
		diffy = tex1Dfetch(tex_y2, idx)-y1;
		diffz = tex1Dfetch(tex_z2, idx)-z1;
		lx = abs(diffx);
		ly = abs(diffy);
		lz = abs(diffz);
		length = ly;
		if (lx > length) {length = lx;}
		if (lz > length) {length = lz;}
		flength = 1.0f / (float)length;
		xinc = diffx * flength;
		yinc = diffy * flength;
		zinc = diffz * flength;
		x = x1;
		y = y1;
		z = z1;
		for (n=0; n<=length; ++n) {
			Qf = Qf + tex1Dfetch(tex_im, (int)z * step + (int)y * nim3 + (int)x);
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
		// compute F
		if (Qf==0.0f) {return;}
		Qf = 1 / Qf;
		Qf = Qf * scale;
		Qi = (unsigned int)Qf;
		x = x1;
		y = y1;
		z = z1;
		for (n=0; n<=length; ++n) {
			atomicAdd(&d_F[(int)z * step + (int)y * nim3 + (int)x], Qi);
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}

	}
}

// OPL-3D-OSEM algorithm with DDA-ELL
void kernel_pet3D_OPLEM_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
							 unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
							 unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
							 float* im, int nim1, int nim2, int nim3,
							 float* NM, int NM1, int NM2, int NM3, int Nsub, int ID){
	
	// Constant according Graphical card
	int mem_max = 800000000; // used only 800 MB on 1 GB
	float scale = 4000.0f;
	float invscale = 1 / scale;
	
	// select a GPU
	if (ID != -1){cudaSetDevice(ID);}

	// vars
	int block_size, grid_size, i;
	dim3 threads, grid;
	int nim = nim1 * nim2 * nim3;
	unsigned short int* d_x1;
	unsigned short int* d_x2;
	unsigned short int* d_y1;
	unsigned short int* d_y2;
	unsigned short int* d_z1;
	unsigned short int* d_z2;
	unsigned int mem_size_point;

	// memory managment 12 B / LORs and 8 B / Vox
	int max_sub, Nouter, mem_sub;
	mem_max -= (12 * nim);
	mem_sub = 12.0f * nx1 / float(Nsub);
	max_sub = int(mem_max / mem_sub);
	Nouter = (Nsub + max_sub - 1) / max_sub;

	printf("max_sub %i   Nouter %i\n", max_sub, Nouter);

	// device mem allocation
	float* d_im;
	unsigned int mem_size_im = nim * sizeof(float);
	cudaMalloc((void**) &d_im, mem_size_im);
	unsigned int* d_F;
	unsigned int mem_size_F = nim * sizeof(int);
	cudaMalloc((void**) &d_F, mem_size_F);
	unsigned int* F = (unsigned int*)malloc(mem_size_F);
	float* d_NM;
	unsigned int mem_size_NM = nim * sizeof(float);
	cudaMalloc((void**) &d_NM, mem_size_NM);
	
	// precompute inplace 1/NM, and load to GPU
	for (i=0; i<nim; ++i) {NM[i] = 1 / NM[i];}
	cudaMemcpy(d_NM, NM, mem_size_NM, cudaMemcpyHostToDevice);

	// Init and update
	cudaMemset(d_F, 0, mem_size_F);
	cudaMemcpy(d_im, im, mem_size_im, cudaMemcpyHostToDevice);

	// Outer loop, avoid memory overflow
	int sub_start, sub_stop, nsub;
	int lor_start, lor_stop, nlor;
	int sublor_start, sublor_stop, nsublor;
	for (int iouter=0; iouter < Nouter; ++iouter) {
		printf("iouter %i\n", iouter);
		// split subset
		sub_start = int(float(Nsub) / Nouter * iouter + 0.5f);
		sub_stop = int(float(Nsub) / Nouter * (iouter+1) + 0.5f);
		nsub = sub_stop - sub_start;
		printf("   sub: %i to %i\n", sub_start, sub_stop);
		// load in memory subsets
		lor_start = int(float(nx1) / Nsub * sub_start + 0.5f);
		lor_stop = int(float(nx1) / Nsub * sub_stop + 0.5f);
		nlor = lor_stop - lor_start;
		printf("   lor: %i to %i\n", lor_start, lor_stop);
		mem_size_point = nlor * sizeof(short int);
		cudaMalloc((void**) &d_x1, mem_size_point);
		cudaMalloc((void**) &d_y1, mem_size_point);
		cudaMalloc((void**) &d_z1, mem_size_point);
		cudaMalloc((void**) &d_x2, mem_size_point);
		cudaMalloc((void**) &d_y2, mem_size_point);
		cudaMalloc((void**) &d_z2, mem_size_point);
		cudaMemcpy(d_x1, &x1[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		cudaMemcpy(d_y1, &y1[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		cudaMemcpy(d_z1, &z1[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		cudaMemcpy(d_x2, &x2[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		cudaMemcpy(d_y2, &y2[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		cudaMemcpy(d_z2, &z2[lor_start], mem_size_point, cudaMemcpyHostToDevice);

		// Bind textures
		cudaBindTexture(NULL, tex_x1, d_x1, mem_size_point);
		cudaBindTexture(NULL, tex_y1, d_y1, mem_size_point);
		cudaBindTexture(NULL, tex_z1, d_z1, mem_size_point);
		cudaBindTexture(NULL, tex_x2, d_x2, mem_size_point);
		cudaBindTexture(NULL, tex_y2, d_y2, mem_size_point);
		cudaBindTexture(NULL, tex_z2, d_z2, mem_size_point);

		// subset loop
		for (int isub=0; isub < nsub; ++isub) {
			// Bind im tex
			cudaBindTexture(NULL, tex_im, d_im, mem_size_im);
			
			// Boundary subset
			sublor_start = int(float(nlor) / nsub * isub + 0.5f);
			sublor_stop = int(float(nlor) / nsub * (isub+1) + 0.5f);
			nsublor = sublor_stop - sublor_start;
			//printf("   isub: %i sublor: %i to %i\n", isub, sublor_start, sublor_stop);

			// Compute corrector subset
			block_size = 256;
			grid_size = (nsublor + block_size - 1) / block_size; // CODE IS LIMITED TO < 16e6 lines
			threads.x = block_size;
			grid.x = grid_size;
			pet3D_OPLEM_DDA<<<grid, threads>>>(d_F, sublor_start, sublor_stop, nim3,
											   nim3*nim3, nim, nsublor, scale);
			// Unbind im tex
			cudaUnbindTexture(tex_im);
			
			// Update the volume
			block_size = 128;
			grid_size = (nim + block_size - 1) / block_size;
			threads.x = block_size;
			grid.x = grid_size;
			pet3D_OPLEM_update<<<grid, threads>>>(d_im, d_F, d_NM, invscale, nim);

		} // isub
		// Unbind textures
		cudaUnbindTexture(tex_x1);
		cudaUnbindTexture(tex_y1);
		cudaUnbindTexture(tex_z1);
		cudaUnbindTexture(tex_x2);
		cudaUnbindTexture(tex_y2);
		cudaUnbindTexture(tex_z2);
		
		// Clean mem
		cudaFree(d_x1);
		cudaFree(d_y1);
		cudaFree(d_z1);
		cudaFree(d_x2);
		cudaFree(d_y2);
		cudaFree(d_z2);
		
	} // iouter

	// Get Back result
	cudaMemcpy(im, d_im, mem_size_im, cudaMemcpyDeviceToHost);

	// Clean mem
	free(F);
	cudaFree(d_im);
	cudaFree(d_F);
	cudaThreadExit();
}

////// Attenuation correction ///////////////////////////////////

// DDA ray-projector with attenuation correction
__global__ void pet3D_OPLEM_DDA_att(unsigned int* d_F, int sublor_start, int sublor_stop, int nim3,
									int step, int nim, int nsublor, float scale) {

	int length, n, diffx, diffy, diffz, ind;
	float flength, x, y, z, lx, ly, lz, xinc, yinc, zinc, Qf, Af;
	unsigned short int x1, y1, z1;
	unsigned int Qi;
	int idx = blockIdx.x * blockDim.x + threadIdx.x + sublor_start;
	
	if (idx < sublor_stop) {
		Qf = 0.0f;
		Af = 0.0f;
		x1 = tex1Dfetch(tex_x1, idx);
		y1 = tex1Dfetch(tex_y1, idx);
		z1 = tex1Dfetch(tex_z1, idx);
		diffx = tex1Dfetch(tex_x2, idx)-x1;
		diffy = tex1Dfetch(tex_y2, idx)-y1;
		diffz = tex1Dfetch(tex_z2, idx)-z1;
		lx = abs(diffx);
		ly = abs(diffy);
		lz = abs(diffz);
		length = ly;
		if (lx > length) {length = lx;}
		if (lz > length) {length = lz;}
		flength = 1.0f / (float)length;
		xinc = diffx * flength;
		yinc = diffy * flength;
		zinc = diffz * flength;
		x = x1;
		y = y1;
		z = z1;
		for (n=0; n<=length; ++n) {
			ind = (int)z * step + (int)y * nim3 + (int)x;
			Qf = Qf + tex1Dfetch(tex_im, ind);
			Af = Af - tex1Dfetch(tex_at, ind);
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}
		// compute F
		if (Qf==0.0f) {return;}
		Qf = Qf * __expf(Af/2.0);
		Qf = 1 / Qf;
		Qf = Qf * scale;
		Qi = (unsigned int)Qf;
		x = x1;
		y = y1;
		z = z1;
		for (n=0; n<=length; ++n) {
			atomicAdd(&d_F[(int)z * step + (int)y * nim3 + (int)x], Qi);
			x = x + xinc;
			y = y + yinc;
			z = z + zinc;
		}

	}
}

// OPL-3D-OSEM algorithm with DDA-ELL with attenuation correction
void kernel_pet3D_OPLEM_att_cuda(unsigned short int* x1, int nx1, unsigned short int* y1, int ny1,
								 unsigned short int* z1, int nz1, unsigned short int* x2, int nx2,
								 unsigned short int* y2, int ny2, unsigned short int* z2, int nz2,
								 float* im, int nim1, int nim2, int nim3,
								 float* NM, int NM1, int NM2, int NM3,
								 float* at, int nat1, int nat2, int nat3,
								 int Nsub, int ID){
	
	// Constant according Graphical card
	int mem_max = 800000000; // used only 800 MB on 1 GB
	float scale = 4000.0f;
	float invscale = 1 / scale;
	
	// select a GPU
	if (ID != -1){cudaSetDevice(ID);}

	// vars
	int block_size, grid_size, i;
	dim3 threads, grid;
	int nim = nim1 * nim2 * nim3;
	unsigned short int* d_x1;
	unsigned short int* d_x2;
	unsigned short int* d_y1;
	unsigned short int* d_y2;
	unsigned short int* d_z1;
	unsigned short int* d_z2;
	unsigned int mem_size_point;

	// memory managment 12 B / LORs and 8 B / Vox
	int max_sub, Nouter, mem_sub;
	mem_max -= (16 * nim);
	mem_sub = 12.0f * nx1 / float(Nsub);
	max_sub = int(mem_max / mem_sub);
	Nouter = (Nsub + max_sub - 1) / max_sub;

	printf("max_sub %i   Nouter %i\n", max_sub, Nouter);

	// device mem allocation
	float* d_im;
	unsigned int mem_size_im = nim * sizeof(float);
	cudaMalloc((void**) &d_im, mem_size_im);
	unsigned int* d_F;
	unsigned int mem_size_F = nim * sizeof(int);
	cudaMalloc((void**) &d_F, mem_size_F);
	unsigned int* F = (unsigned int*)malloc(mem_size_F);
	float* d_NM;
	unsigned int mem_size_NM = nim * sizeof(float);
	cudaMalloc((void**) &d_NM, mem_size_NM);
	float* d_at;
	unsigned int mem_size_at = nim * sizeof(float);
	cudaMalloc((void**) &d_at, mem_size_at);
	
	// precompute inplace 1/NM, and load to GPU
	for (i=0; i<nim; ++i) {NM[i] = 1 / NM[i];}
	cudaMemcpy(d_NM, NM, mem_size_NM, cudaMemcpyHostToDevice);

	// Init and update
	cudaMemset(d_F, 0, mem_size_F);
	cudaMemcpy(d_im, im, mem_size_im, cudaMemcpyHostToDevice);
	cudaMemcpy(d_at, at, mem_size_at, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_at, d_at, mem_size_at);

	// Outer loop, avoid memory overflow
	int sub_start, sub_stop, nsub;
	int lor_start, lor_stop, nlor;
	int sublor_start, sublor_stop, nsublor;
	for (int iouter=0; iouter < Nouter; ++iouter) {
		printf("iouter %i\n", iouter);
		// split subset
		sub_start = int(float(Nsub) / Nouter * iouter + 0.5f);
		sub_stop = int(float(Nsub) / Nouter * (iouter+1) + 0.5f);
		nsub = sub_stop - sub_start;
		printf("   sub: %i to %i\n", sub_start, sub_stop);
		// load in memory subsets
		lor_start = int(float(nx1) / Nsub * sub_start + 0.5f);
		lor_stop = int(float(nx1) / Nsub * sub_stop + 0.5f);
		nlor = lor_stop - lor_start;
		printf("   lor: %i to %i\n", lor_start, lor_stop);
		mem_size_point = nlor * sizeof(short int);
		cudaMalloc((void**) &d_x1, mem_size_point);
		cudaMalloc((void**) &d_y1, mem_size_point);
		cudaMalloc((void**) &d_z1, mem_size_point);
		cudaMalloc((void**) &d_x2, mem_size_point);
		cudaMalloc((void**) &d_y2, mem_size_point);
		cudaMalloc((void**) &d_z2, mem_size_point);
		cudaMemcpy(d_x1, &x1[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		cudaMemcpy(d_y1, &y1[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		cudaMemcpy(d_z1, &z1[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		cudaMemcpy(d_x2, &x2[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		cudaMemcpy(d_y2, &y2[lor_start], mem_size_point, cudaMemcpyHostToDevice);
		cudaMemcpy(d_z2, &z2[lor_start], mem_size_point, cudaMemcpyHostToDevice);

		// Bind textures
		cudaBindTexture(NULL, tex_x1, d_x1, mem_size_point);
		cudaBindTexture(NULL, tex_y1, d_y1, mem_size_point);
		cudaBindTexture(NULL, tex_z1, d_z1, mem_size_point);
		cudaBindTexture(NULL, tex_x2, d_x2, mem_size_point);
		cudaBindTexture(NULL, tex_y2, d_y2, mem_size_point);
		cudaBindTexture(NULL, tex_z2, d_z2, mem_size_point);

		// subset loop
		for (int isub=0; isub < nsub; ++isub) {
			// Bind im tex
			cudaBindTexture(NULL, tex_im, d_im, mem_size_im);
			
			// Boundary subset
			sublor_start = int(float(nlor) / nsub * isub + 0.5f);
			sublor_stop = int(float(nlor) / nsub * (isub+1) + 0.5f);
			nsublor = sublor_stop - sublor_start;
			//printf("   isub: %i sublor: %i to %i\n", isub, sublor_start, sublor_stop);

			// Compute corrector subset
			block_size = 256;
			grid_size = (nsublor + block_size - 1) / block_size; // CODE IS LIMITED TO < 16e6 lines
			threads.x = block_size;
			grid.x = grid_size;
			pet3D_OPLEM_DDA_att<<<grid, threads>>>(d_F, sublor_start, sublor_stop, nim3,
												   nim3*nim3, nim, nsublor, scale);
			// Unbind im tex
			cudaUnbindTexture(tex_im);
			
			// Update the volume
			block_size = 128;
			grid_size = (nim + block_size - 1) / block_size;
			threads.x = block_size;
			grid.x = grid_size;
			pet3D_OPLEM_update<<<grid, threads>>>(d_im, d_F, d_NM, invscale, nim);

		} // isub
		// Unbind textures
		cudaUnbindTexture(tex_x1);
		cudaUnbindTexture(tex_y1);
		cudaUnbindTexture(tex_z1);
		cudaUnbindTexture(tex_x2);
		cudaUnbindTexture(tex_y2);
		cudaUnbindTexture(tex_z2);
		
		// Clean mem
		cudaFree(d_x1);
		cudaFree(d_y1);
		cudaFree(d_z1);
		cudaFree(d_x2);
		cudaFree(d_y2);
		cudaFree(d_z2);
		
	} // iouter

	// Get Back result
	cudaMemcpy(im, d_im, mem_size_im, cudaMemcpyDeviceToHost);

	// Clean mem
	cudaUnbindTexture(tex_at);
	free(F);
	cudaFree(d_at);
	cudaFree(d_im);
	cudaFree(d_F);
	cudaThreadExit();
}
