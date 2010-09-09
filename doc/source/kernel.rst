Kernel
======

kernel_3Dconv_cuda
-------------------

**kernel_3Dconv_cuda** (vol, H)

*Perform the convolution between a volume and a 3D transfert function with the GPU. The convolution is done in Fourrier space with cuda and cufft.*

**Parameters**

``vol`` 3D Numpy array to be convolued, the dimension must all the same (nz=ny=nx) in order to used the FFT. If it's not the case, please use the function *volume_pack_cube* in order to pack your volume into a cube. The convolution is made in-place, so the volume vol will be overwritten.

``H`` 3D transfert function, must be a Numpy array and prepare to be used with cufft. To do that use the function *filter_pad_3d_cuda*, in order to prepare properly the filter.

**Returns**

**Notes**

If you want get back the original size of your volume, in the case that you used previously the function *volume_pack_cube*, you can use this function *volume_unpack_cube*.

**Examples**

::

	>>> vol = volume_open('myvol.vol')
	>>> nz, ny, nx = vol.shape
	>>> cube = volume_pack_cube(vol)
	>>> H    = filter_build_3d_Metz(141, 2, 0.2)
	>>> Hpad = filter_pad_3d_cuda(H)
	>>> kernel_3Dconv_cuda(cube, Hpad)
	>>> vol  = volume_unpack_cube(cube, nz, ny, nx)
