#include "kernel_cuda.h"
#include <stdio.h>
#include <cublas.h>

void kernel_pet2D_EMML_wrap_cuda(float* SRM, int nlor, int npix, float* im, int npixim, int* LOR_val, int nval) {
	// select a GPU
	cudaSetDevice(0);
	// init cublas
	cublasStatus status;
	status = cublasInit();
	// allocate host memory for Q and F
	int size_Q = nlor * sizeof(float);
	float* h_Q = (float*)malloc(size_Q);
	float* h_F = (float*)malloc(size_Q);
	// allocate device memory for SRM, im, Q and F
	int size_SRM = nlor * npix;
	float* d_SRM;
	float* d_im;
	float* d_Q;
	float* d_F;
	status = cublasAlloc(size_SRM, sizeof(float), (void**)&d_SRM);
	status = cublasAlloc(npixim, sizeof(float), (void**)&d_im);
	status = cublasAlloc(nlor, sizeof(float), (void**)&d_Q);
	status = cublasAlloc(nlor, sizeof(float), (void**)&d_F);
	// load SRM and im to the device
	status = cublasSetVector(size_SRM, sizeof(float), SRM, 1, d_SRM, 1);
	status = cublasSetVector(npixim, sizeof(float), im, 1, d_im, 1);
	// clear the last error
	cublasGetError();
	// compute Q
	cublasSgemv('t', npix, nlor, 1.0, d_SRM, npix, d_im, 1, 0.0, d_Q, 1);
	status = cublasGetError();
	status = cublasGetVector(nlor, sizeof(float), d_Q, 1, h_Q, 1);

	// check
	int i, j;
	float* F=(float*)malloc(nlor*sizeof(float));
	float f;
	for (j=0;j<npix;++j) {
		f = 0.0;
		for (i=0; i<nlor; ++i) {
			f += ((float)LOR_val[i] * SRM[i*npix+j] / h_Q[i]);
			//f = ((float)LOR_val[i] / h_Q[i]);
			//printf("f %f\n", f);
			//printf("LOR_val %f SRM %f\n", (float)LOR_val[i], SRM[i*npix+300]);
			F[j] = f;
		}
	}
	
	// compute qi = LOR_val / qi
	//int j;
	for (j=0; j<nlor; ++j) {h_Q[j] = (float)LOR_val[j] / h_Q[j];}
	//for (i=0;i<nlor;++i){printf("f %f\n", h_Q[i]);}
	// compute f = sum{SRMi / qi} for each i LOR
	status = cublasSetVector(nlor, sizeof(float), h_Q, 1, d_Q, 1);
	cublasGetError();
	cublasSgemv('n', npix, nlor, 1.0, d_SRM, npix, d_Q, 1, 0.0, d_F, 1);
	status = cublasGetError();
	status = cublasGetVector(nlor, sizeof(float), d_F, 1, h_F, 1);

	/*
	int sizei = 10;
	float* hA;
	float* hB;
	float* hC;
	hA = (float*)malloc(sizei * sizeof(float));
	hB = (float*)malloc(2 * sizeof(float));
	hC = (float*)malloc(5 * sizeof(float));
	for (i=0;i<5;++i){hA[i] = 1.0;}
	for (i=5;i<10;++i){hA[i] = 2.0;}
	hB[0] = 1.0;
	hB[1] = 1.0;
	float* dA;
	float* dB;
	float* dC;
	status = cublasAlloc(sizei, sizeof(float), (void**)&dA);
	status = cublasAlloc(2, sizeof(float), (void**)&dB);
	status = cublasAlloc(5, sizeof(float), (void**)&dC);
	status = cublasSetVector(sizei, sizeof(float), hA, 1, dA, 1);
	status = cublasSetVector(2, sizeof(float), hB, 1, dB, 1);
	status = cublasSetVector(5, sizeof(float), hC, 1, dC, 1);
	cublasGetError();
	cublasSgemv('n', 5, 2, 1.0, dA, 5, dB, 1, 0.0, dC, 1);
	status = cublasGetError();
	status = cublasGetVector(5, sizeof(float), dC, 1, hC, 1);
	printf("%f %f %f %f %f\n", hC[0], hC[1], hC[2], hC[3], hC[4]);
	
	*/
	
	for(i=0;i<nlor;++i){
		printf("cpu %f gpu %f\n", F[i], h_F[i]);
	}
	
	// free memory
	status = cublasFree(d_SRM);
	status = cublasFree(d_im);
	// prepare to quit
	status = cublasShutdown();
	

}
