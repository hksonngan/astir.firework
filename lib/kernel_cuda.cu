#include "kernel_cuda.h"
#include <stdio.h>
#include <cublas.h>

// kernel to update image in pet2D EMML algorithm
__global__ void pet2D_im_update(float* im, float* S, float* F, int npix) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < npix && F[idx] != 0.0f) {
		im[idx] = im[idx] / S[idx] * F[idx];
	}
}
// kernel to update Q value in pet2D EMML algorithm
__global__ void pet2D_Q_update(int* d_lorval, float* d_Q, int nval) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nval) {
		d_Q[idx] = (float)d_lorval[idx] / d_Q[idx];
	}
}

void kernel_pet2D_EMML_wrap_cuda(float* SRM, int nlor, int npix, float* im, int npixim, int* LOR_val, int nval, float* S, int ns, int maxit) {
	// select a GPU
	cudaSetDevice(0);
	// init cublas
	cublasStatus status;
	status = cublasInit();
	// allocate device memory for SRM, im, Q and F
	int size_SRM = nlor * npix;
	float* d_SRM;
	float* d_im;
	float* d_Q;
	float* d_F;
	float* d_S;
	int* d_lorval;
	status = cublasAlloc(size_SRM, sizeof(float), (void**)&d_SRM);
	status = cublasAlloc(npixim, sizeof(float), (void**)&d_im);
	status = cublasAlloc(nlor, sizeof(float), (void**)&d_Q);
	status = cublasAlloc(nlor, sizeof(float), (void**)&d_F);
	status = cublasAlloc(ns, sizeof(float), (void**)&d_S);
	status = cublasAlloc(nval, sizeof(int), (void**)&d_lorval);
	// load SRM, SM, LOR_val and im to the device
	status = cublasSetVector(size_SRM, sizeof(float), SRM, 1, d_SRM, 1);
	status = cublasSetVector(npixim, sizeof(float), im, 1, d_im, 1);
	status = cublasSetVector(ns, sizeof(float), S, 1, d_S, 1);
	status = cublasSetVector(nval, sizeof(int), LOR_val, 1, d_lorval, 1);
	if (status != 0) {exit(0);}
	int ite, block_size1, grid_size1, block_size2, grid_size2;
	block_size1 = 512;
	grid_size1 = (nlor + block_size1 - 1) / block_size1;
	block_size2 = 64;
	grid_size2 = (npix + block_size2 - 1) / block_size2;
	dim3 threads1(block_size1);
	dim3 grid1(grid_size1);
	dim3 threads2(block_size2);
	dim3 grid2(grid_size2);
	for (ite=0; ite<maxit; ++ite) {
		// compute Q
		cublasSgemv('t', npix, nlor, 1.0, d_SRM, npix, d_im, 1, 0.0, d_Q, 1);
		pet2D_Q_update<<< grid1, threads1 >>>(d_lorval, d_Q, nval);
		// compute f = sum{SRMi / qi} for each i LOR
		cublasSgemv('n', npix, nlor, 1.0, d_SRM, npix, d_Q, 1, 0.0, d_F, 1);
		// update image
		pet2D_im_update<<< grid2, threads2 >>>(d_im, d_S, d_F, npix);
	}
	// get results
	status = cublasGetError();
	status = cublasGetVector(npix, sizeof(float), d_im, 1, im, 1);
	// free memory
	status = cublasFree(d_SRM);
	status = cublasFree(d_im);
	status = cublasFree(d_Q);
	status = cublasFree(d_F);
	status = cublasFree(d_S);
	// prepare to quit
	status = cublasShutdown();
}
