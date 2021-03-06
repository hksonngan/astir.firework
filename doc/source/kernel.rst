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

kernel_filter_2d_adaptive_median
--------------------------------

**kernel_filter_2d_median** (im_in, im_out, w, wmax)

*Proceed a 2D adaptive median filter on image. It's like median filter apllied with servelas kernel size. The main advantage of the adaptive version it is that the high frequencies are less cutoff than the classical one.*

**Parameters**

``im_in`` A 2D Numpy array as image ('float32'), it is the input image to be filtered.

``im_out`` A 2D Numpy array as return image ('float32'), with every values set to zero. This array must be the same size as *im_in*.

``w`` First size of the kernel, must be odd.

``wmax`` Is the last size of the kernel, must be odd.

**Returns**

``im_out`` The image is proceed inplace, so this 2D Numpy array will be overwritten by the result of the filtration.

**Examples**

::

	>>> im = image_open('test.png')
	>>> res = zeros(im.shape, 'float32')
	>>> kernel_filter_2d_adaptive_median(im, res, 3, 9)
	>>> image_show(res)

kernel_filter_2d_median
-----------------------

**kernel_filter_2d_median** (im_in, im_out, w)

*Proceed a 2D median filter on image*

**Parameters**

``im_in`` A 2D Numpy array as image ('float32'), it is the input image to be filtered.

``im_out`` A 2D Numpy array as return image ('float32'), with every values set to zero. This array must be the same size as *im_in*.

``w`` Size of the filter window, must be odd.

**Returns**

``im_out`` The image is proceed inplace, so this 2D Numpy array will be overwritten by the result of the filtration.

**Examples**

::

	>>> im = image_open('test.png')
	>>> res = zeros(im.shape, 'float32')
	>>> kernel_filter_2d_median(im, res, 5)
	>>> image_show(res)


kernel_filter_3d_adaptive_median
--------------------------------

**kernel_filter_3d_adaptive_median** (vol_in, vol_out, w, wmax)

*Proceed a 3D adaptive median filter on image. It's like median filter apllied with servelas kernel size. The main advantage of the adaptive version it is that the high frequencies are less cutoff than the classical one.*

**Parameters**

``vol_in`` A 3D Numpy array as image ('float32'), it is the input image to be filtered.

``vol_out`` A 3D Numpy array as return image ('float32'), with every values set to zero. This array must be the same size as *im_in*.

``w`` First size of the kernel, must be odd.

``wmax`` Is the last size of the kernel, must be odd.

**Returns**

``vol_out`` The volume is proceed inplace, so this 3D Numpy array will be overwritten by the result of the filtration.

**Examples**

::

	>>> vol = volume_open('test.vol')
	>>> res = zeros(vol.shape, 'float32')
	>>> kernel_filter_3d_adaptive_median(vol, res, 3, 9)
	>>> volume_write(res, 'res.vol')

	
kernel_filter_3d_median
-----------------------

**kernel_filter_3d_median** (vol_in, vol_out, w)

*Proceed a 3D median filter on volume*

**Parameters**

``vol_in`` A 3D Numpy array as image ('float32'), it is the input volume to be filtered.

``vol_out`` A 3D Numpy array as return volume ('float32'), with every values set to zero. This array must be the same size as *vol_in*.

``w`` Size of the filter window, must be odd.

**Returns**

``vol_out`` The volume is proceed inplace, so this 3D Numpy array will be overwritten by the result of the filtration.

**Examples**

::

	>>> vol = volume_open('test.vol')
	>>> res = zeros(vol.shape, 'float32')
	>>> kernel_filter_3d_median(vol, res, 5)

kernel_mip_volume_rendering
---------------------------

**kernel_mip_volume_rendering** (vol, im_mip, phy, theta, scale)

*Compute the mip (Maximum Intensity Projection) image of a volume for any angle view defined by the two Eulerian angles phy and theta. The mip image is computed by raycasting.*

**Parameters**

``vol`` The input volume (3D Numpy array).

``im_mip`` An empty image in order to collect the mip, its size define the filed-of-view, thus the number of rays used to compute the mip.

``phy, theta`` The two Eulerian angles that define the angle view.

``scale`` The up scale factor, in order to change the magnification of the volume on the mip image.

**Returns**

``im_mip`` The mip image of the volume according the angle view.

	
kernel_resampling_2d_Lanczos2
-----------------------------

**kernel_resampling_3d_Lanczos2** (im_in, im_out)

*Proceed a Lanczos2 resampling on image*

**Parameters**

``im_in`` Original image (2D Numpy array).

``im_out`` An empty image with the new dimensions required.

**Returns**

``im_out`` The image is resampled inplace according its dimensions.

	
kernel_resampling_3d_Lanczos2
-----------------------------

**kernel_resampling_3d_Lanczos2** (vol_in, vol_out)

*Proceed a Lanczos2 resampling on volume*

**Parameters**

``vol_in`` Original volume (3D Numpy array).

``vol_out`` An empty volume with the new dimensions required.

**Returns**

``vol_out`` The volume is resampled inplace according its dimensions.

	
kernel_resampling_3d_Lanczos3
-----------------------------

**kernel_resampling_3d_Lanczos3** (vol_in, vol_out)

*Proceed a Lanczos3 resampling on volume*

**Parameters**

``vol_in`` Original volume (3D Numpy array).

``vol_out`` An empty volume with the new dimensions required.

**Returns**

``vol_out`` The volume is resampled inplace according its dimensions.
