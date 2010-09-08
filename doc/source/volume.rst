Volume
======

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

