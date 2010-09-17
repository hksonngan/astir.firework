Volume
======

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

``file_name`` Name of the file which contains the volume in *.vol* format (FIREwork format)

**Returns**

``newvol`` Return a numpy array with dimension *(nz, ny, nx)* and *float32* format.

**Notes**

**Examples**

::

	>>> vol = volume_open('volume.vol')

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

volume_slice
------------

im = **volume_slice** (vol, pos, [axe])

*Return the slice image from a volume according the position and the axe of the slice.*

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
