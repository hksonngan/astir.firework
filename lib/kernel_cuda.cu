#include "kernel_cuda.h"
#include <stdio.h>
#include <cublas.h>
#include <cufft.h>
#include <sys/time.h>
#include <math_constants.h>

// Perform a multiplication between a complex and a real vectors
__global__ void vector_complex_x_real(cufftComplex* dcpx, float* dr, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float r, j, h;
	if (idx < n) {
		r = dcpx[idx].x;
		j = dcpx[idx].y;
		h = dr[idx];
		r = r * h;
		j = j * h;
		dcpx[idx].x = r;
		dcpx[idx].y = j;
	}
}

// Perform a mulitplication between a real vectors and an alpha value
__global__ void vector_real_x_cst(float* dr, float alpha, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float val;
	if (idx < n) {
		val = dr[idx];
		val = val * alpha;
		dr[idx] = val;
	}
}

// 3D convolution (in Fourier)
void kernel_3D_conv_wrap_cuda(float* vol, int nz, int ny, int nx, float* H, int a, int b, int c) {
	int ID = 0;
	// select a GPU
	if (ID != -1){cudaSetDevice(ID);}
	// prepare the filter
	int nc = (ny / 2) + 1;
	int size_H = c * b * a;
	int size_vol = nz * ny * nx;
	int size_fft = nz * nc * nx;
	
	cufftHandle plan_forward, plan_inverse;
	cufftReal* dvol;
	cufftComplex* dfft;
	float* dH;

	// alloc mem GPU
	cudaMalloc((void**)&dvol, size_vol * sizeof(cufftReal));
	//printf("dvol %i\n", status);
	cudaMalloc((void**)&dfft, size_fft * sizeof(cufftComplex));
	//printf("dfft %i\n", status);
	cudaMalloc((void**)&dH, size_H * sizeof(float));
	//printf("dH %i\n", status);
	
	// tranfert to GPU
	cudaMemcpy(dvol, vol, size_vol * sizeof(cufftReal), cudaMemcpyHostToDevice);
	//printf("memcpy dvol %i\n", status);
	cudaMemcpy(dH, H, size_H * sizeof(float), cudaMemcpyHostToDevice);
	//printf("memcpy dH %i\n", status);
	
	// do fft
	cufftPlan3d(&plan_forward, nx, ny, nz, CUFFT_R2C);
	//printf("init plan %i\n", status);
	cufftExecR2C(plan_forward, dvol, dfft);
	//printf("fft %i\n", status);
	
	// do 3D convolution
	int block_size, grid_size;
	dim3 threads, grid;
	block_size = 128;
	grid_size = (size_fft + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	vector_complex_x_real<<<grid, threads>>>(dfft, dH, size_fft);

	// get inverse transform
	cufftPlan3d(&plan_inverse, nz, ny, nx, CUFFT_C2R);
	cufftExecC2R(plan_inverse, dfft, dvol);

	// Normalize values due to FFT theorem (1 / N)
	block_size = 128;
	grid_size = (size_vol + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	vector_real_x_cst<<<grid, threads>>>(dvol, 1 / float(size_vol), size_vol);

	// get back the volume
	cudaMemcpy(vol, dvol, size_vol * sizeof(float), cudaMemcpyDeviceToHost);
	
	// clean up
	cufftDestroy(plan_forward);
	cufftDestroy(plan_inverse);
	cudaFree(dvol);
	cudaFree(dH);
	cudaFree(dfft);
	
	cudaThreadExit();
}

