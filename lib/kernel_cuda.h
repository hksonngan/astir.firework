void kernel_pet2D_EMML_wrap_cuda(float* SRM, int nlor, int npix, float* im, int npixim, int* LOR_val, int nval, float* S, int ns, int maxit);
void kernel_pet2D_SRM_DDA_wrap_cuda(float* SRM, int wy, int wx, int* X1, int nx1, int* Y1, int ny1, int* X2, int nx2, int* Y2, int ny2, int width_image);


