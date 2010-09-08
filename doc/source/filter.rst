Filter
======

filter_build_3d_Butterworth_lp
------------------------------

H = **filter_build_3d_Butterworth_lp** (size, N, fc)

*Build a 3D lowpass Butterworth filter*

**Parameters**

``size`` Size of the edge of the 3D transfert function.

``N`` Order of the filter. 

``fc`` Cut-off frequency.

**Returns**

``H`` Numpy array of 3 dimensions *(size, size, size)*, which contains coefficients of the transfert function.

**Notes**

**Examples**

::

	>>> H = filter_build_3d_Butterworth_lp(141, 2, 0.2)


filter_build_3d_Gaussian
------------------------

H = **filter_build_3d_Gaussian** (size, fc)

*Build a 3D Gaussian filter*

**Parameters**

``size`` Size of the edge of the 3D transfert function.

``fc`` Cut-off frequency, equivalent to the sigma value.

**Returns**

``H`` Numpy array of 3 dimensions *(size, size, size)*, which contains coefficients of the transfert function.

**Notes**

**Examples**

::

	>>> H = filter_build_3d_Gaussian(141, 0.2)
	

filter_build_3d_Metz
--------------------

H = **filter_build_3d_Metz** (size, N, fc)

*Build a 3D Metz filter*

**Parameters**

``size`` Size of the edge of the 3D transfert function.

``N`` Order of the filter. If *N=0* the filter is equivalent to a Gaussian filter. More *N>0* more the filter has gain in medium frequencies.

``fc`` Cut-off frequency, equivalent to sigma of the Gaussian filter.

**Returns**

``H`` Numpy array of 3 dimensions *(size, size, size)*, which contains coefficients of the transfert function.

**Notes**

**Examples**

::

	>>> H = filter_build_3d_Metz(141, 2, 0.2)


filter_conv_3d_cuda
-------------------

volf = **filter_conv_3d_cuda** (vol, H)

*Perform the convolution between a volume and a 3D transfert function with the GPU. The convolution is done in Fourrier space with cuda and cufft.*

**Parameters**

``vol`` 3D Numpy array to be convolued, the dimension must all the same (nz=ny=nx) in order to used the FFT. If it's not the case, please use the function *volume_pack_cube* in order to pack your volume into a cube.

``H`` 3D transfert function, must be a Numpy array and prepare to be used with cufft. To do that use the function *filter_pad_3d_cuda*, in order to prepare properly the filter.

**Returns**

``volf`` Numpy array of 3 dimensions after the function transfert applied. If you want get back the original size of your volume, in the case that you used previously the function *volume_pack_cube*, you can use this function *volume_unpack_cube*.

**Notes**

**Examples**

::

	>>> vol = volume_open('myvol.vol')
	>>> nz, ny, nx = vol.shape
	>>> cube = volume_pack_cube(vol)
	>>> H    = filter_build_3d_Metz(141, 2, 0.2)
	>>> Hpad = filter_pad_3d_cuda(H)
	>>> volf = filter_conv_3d_cuda(vol, Hpad)
	>>> vol  = volume_unpack_cube(volf, nz, ny, nx)

	
filter_pad_3d_cuda
------------------

Hpad = **filter_pad_3d_cuda** (H)

*Shift, pad and crop a 3d filter in order to be used by a convolution perform by cuda (cufft). This function well prepare the filter according the FFT format provide by cuda (non redundant coefficients).*

**Parameters**

``H`` 3D transfert function, must be a Numpy array.

**Returns**

``Hpad`` Numpy array of 3 dimensions. 

**Notes**

**Examples**

::

	>>> H = filter_build_3d_Metz(141, 2, 0.2)
	>>> Hpad = filter_pad_3d_cuda(H)

