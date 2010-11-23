Volume
======

volume_fft
----------

VOL = **volume_fft** (vol)

*Compute the 3D FFT of volume*

**Parameters**

``vol`` 3D Numpy array as volume, must be cube dimension and in 'float32' format

**Returns**

``VOL`` 3D Numpy array of FFT complexe values ('complex128'). The spectrum is already shift.

**Notes**

In order to pack the volume to a center of an empty cube use the function volume_pack_cube.

volume_fsc
----------

fsc, freq = **volume_fsc** (vol1, vol2)

*Compute the Fourier Shell Correlation between two volumes*

**Parameters**

``vol1, vol2`` Two 3D Numpy array as volumes, must be cube volume.

**Returns**

``fsc`` 1D array of fsc values

``freq`` 1D array of Nyquist frequencies for each fsc values

**Notes**

Input volumes must be not normalize before computing the fsc, in order to avoid (negative value).

volume_ifft
-----------

vol = **volume_ifft* (VOL)

*Compute the inverse 3D FFT of a 3D volume spectrum*

**Parameters**

``VOL`` 3D Numpy array as shift volume spectrum ('complexe128' format), must be cube dimension.

**Returns**

``vol`` 3D Numpy array as volume in 'float32' format

volume_infos
------------

**volume_infos** (vol)

*Display some usefull informations about a volume (dimensions, min, max, mean, and std)*

**Parameters**

``vol`` 3D Numpy array as volume.

volume_mask_box
---------------

mask = *volume_mask_box* (nz, ny, nx, w, h, d)

*Create a 3D box mask*

**Paramters**

``nz, ny, nx`` size of the 3D Numpy array which contains the mask

``w, h, d`` Width, height and depth of the mask centered to the volume

**Returns**

``mask`` Return the mask as 3D Numpy array

volume_mask_cylinder
--------------------

mask = *volume_mask_cylinder* (nz, ny, nx, h, rad)

*Create a 3D cylinder mask*

**Paramters**

``nz, ny, nx`` size of the 3D Numpy array which contains the mask

``h, rad`` height and radius of cylinder centered to the volume

**Returns**

``mask`` Return the mask as 3D Numpy array

volume_mip
----------

immip = **volume_mip** (volume_name, axe='z')

*Return the Maximum Intensity Projection (MIP) of a volume.*

**Parameters**

``volume_name`` Volume name, must be a numpy array with 3 dimensions.

``axe`` Axis of the MIP, by default is 'z' (coronal MIP), can be 'x' (transversal MIP), or 'y' (sagital MIP)

**Returns**

``immip`` Return MIP's image as Numpy array of dimension *(ny, nx)*.

**Notes**

Axis according 'x' and 'y' must be verify!!

**Examples**

::

	>>> vol = volume_open('volume.vol')
	>>> im  = volume_mip(vol)
	>>> image_show(im)

volume_miip
-----------

immiip = **volume_miip** (volume_name, axe='z')

*Return the Minimum Intensity Projection (MiIP) of a volume.*

**Parameters**

``volume_name`` Volume name, must be a numpy array with 3 dimensions.

``axe`` Axis of the MiIP, by default is 'z' (coronal MiIP), can be 'x' (transversal MiIP), or 'y' (sagital MiIP)

**Returns**

``immiip`` Return MiIP's image as Numpy array of dimension *(ny, nx)*.

**Notes**

Axis according 'x' and 'y' must be verify!!

	
volume_mosaic
-------------

mos = **volume_mosaic** (vol, [axe], [norm])

*Create a mosaic image from a volume, very usefull to compare each volume slice on one image.*

**Parameters**

``vol`` 3D Numpy array as volume

``axe`` Axis of slice to build the mosaic, default is 'z' (transversal view)

``norm`` Specifie if each slice are normalize separetly, default is False

**Returns**

``mos`` Mosaic of images which contains slice for every axis value (2D Numpy array)

**Examples**

::

	>>> vol = volume_open('test.vol')
	>>> mos = volume_mosaic(vol, norm=True)
	>>> image_write_mapcolor(mos, 'mos.png')

volume_open
-----------

newvol = **volume_open** (file_name)

*Open a volume file.*

**Parameters**

``file_name`` Name of the file which contains the volume in FIREwork format. The extension must be *.vol*.

**Returns**

``newvol`` Return a numpy array with dimension *(nz, ny, nx)* and *float32* format.

**Notes**

**Examples**

::

	>>> vol = volume_open('volume.vol')

volume_pack_center
------------------

newvol = **volume_pack_center** (vol, newz, newy, newx)

*Pack a volume to a new one at the center position*

**Parameters**

``vol`` Volume to be packing, must be a 3D Numpy array.

``newz, newy, newx`` New dimension of the volume

**Returns**

``newvol`` 3D Numpy array as volume.
	
volume_pack_cube
----------------

newvol = **volume_pack_cube** (vol)

*Pack a volume with different dimensions (ex. nz=45, ny=141 and nx=141) centered inside a cube (nz=ny=nx). It's usefull when you need to perform FFT on a volume which has different dimensions. This function copy the old volume to the center at a new one (in this case nz=141, ny=141 and nx=141)* 

**Parameters**

``vol`` Volume name, must be Numpy array of 3 dimensions.

**Returns**

``newvol`` New Numpy array, wich contains the volume centered in a cube.

**Notes**

**Examples**

::

	>>> cube = volume_pack_cube(vol)

volume_pows
-----------

pows = **volume_pows** (vol)

*Compute the 3D Power Spectrum of a volume*

**Parameters**

``vol`` 3D Numpy array as volume, must be a cube volume.

**Returns**

``pows`` The 3D power spectrum in a 3D Numpy array.

volume_projection
-----------------

im = **volume_projection** (vol, [axis])

*Compute the 2D projection of a volume along the specified axis*

**Parameters**

``vol`` A 3D Numpy array as volume

``axis`` Axis of the projection can be 'x', 'y' or 'z' (default is 'z')

**Returns**

``im`` The image projection in 2D Numpy array format

volume_ra
---------

val = **volume_ra** (vol)

*Compute the Radial Average of a volume*

**Parameters**

``vol`` A 3D Numpy array as volume

**Returns**

``val`` 1D array, which contains the values of RA

volume_raps
-----------

val, freq = **volume_raps** (vol)

*Compute the Radial Averaging Power Spectrum from a volume*

**Parameters**

``vol`` A 3D Numpy array as volume ('float32')

**Returns**

``val`` 1D array, which contains the values of the RAPS

``freq`` 1D array of Nyquist frequencies for each values of RAPS

**Notes**

The input volume is not normalize i.e. the mean is not equal to zeros. The input must be a cube volume.

volume_raw_open
---------------

vol = **volume_raw_open** (filename, nz, ny, nx, dataformat)

*Open a binary file which contains a volume and convert it to Numpy 3D array*

**Parameters**

``filename`` Filename of the binary file

``nz, ny, nx`` Dimensions of the volume in order to convert it to 3D Numpy array

``dataformat`` Data type must be specified to Numpy, and depend of your binary file, can be 'uint8', 'uint16', 'int32', 'float32', etc.

**Returns**

``vol`` A 3D Numpy array as volume

volume_raw_write
----------------

**volume_raw_write** (vol, filename)

*Write the volume in binary format, the number of byte is define by the type of the Numpy array*

**Parameters**

``vol`` A 3D Numpy array as volume

``name`` Filename to export the volume


volume_slice
------------

im = **volume_slice** (vol, pos, [axe])

*Return the slice image from a volume according the position and the slice axis.*

**Parameters**

``vol`` Volume name, must be a 3D Numpy array.

``pos`` Position of the slice inside the volume.

``axe`` Axe of the slice 'x', 'y', or 'z', by default the value is set to 'z', which is the transversal axis.

**Returns**

``im`` Image of the slice, 2D Numpy array.

**Examples**

::

	>>> vol = volume_open('test.vol')
	>>> im  = volume_slice(vol, 22)
	>>> image_show(im)

volume_unpack_cube
------------------

newvol = **volume_unpack_cube** (vol, oz, oy, ox)

*Crop a cube dimension volume to its original dimensions. Most of time is applied after using the function volume_unpack_cube to get back the original volume.*

**Parameters**

``vol`` 3D Numpy array input volume

``oz, oy, ox`` Original dimensions of the volume before packing it into a cube.

**Returns**

``newvol`` Volume cropped (3D Numpy array)
	
volume_write
------------

**volume_write** (vol, filename)

*Export the volume in FIREwork format (.vol)*

**Parameters**

``vol`` A 3D Numpy array as volume (will convert in float by the function)

``name`` Filename to export the volume, the extension must be *.vol*
