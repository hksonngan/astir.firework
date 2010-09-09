Image
=====

image_normalize
---------------

imnorm = **image_normalize** (im)

*Normalize an image, zeros means and unit standard deviation.*

**Parameters**

``im`` Image name, must be a Numpy array of 2 dimensions.

**Returns**

``imnorm`` Image normalized (Numpy array)

**Notes**

utils.py

**Examples**

::

	>>> im = image_normalize(im)


image_open
-----------

image = **image_open** (filename)

*Load an image as a 2D Numpy array*

**Parameters**

``filename`` Name of the file you want read. Different kind of format is supported like *bmp*, *png*, *tif*, *jpg* and *im* which is the FIREwork image format.

**Returns**

``image`` A 2D Numpy array, the values format is *float32*.

**Notes**

If file contains more one channel, it will be convert in luminance format.

**Examples**

::

	>>> im = image_open('test.png')
	>>> im = image_open('test.im')


	
image_show
----------

**image_show** (im, mapcolor)

*Display an image*

**Parameters**

``im`` Image name, must be a Numpy array of 2 dimensions.

``mapcolor`` Image is display with different kind of colormaps, like *jet*, *hot* and *hsv*, by default it's *jet*.

**Returns**

**Notes**

viewer.py

**Examples**

::

	>>> im = image_open('test.png')
	>>> image_show(im)


image_write
-----------

**image_write** (imagename, filename)

*Save a 2D Numpy array as an image*

**Parameters**

``imagename`` Name of 2D Numpy array. The value format must be in *float32*.

``filename`` Name of the file you want to export the image. Different kind of format is supported like *bmp*, *png*, *tif*, *jpg* and *im* which is the FIREwork image format.

**Returns**

**Notes**

All images saved must have only one channel, i.e. luminance values. If you want export an image with a colormap use the function *image_write_mapcolor*.

In the case you export an image in *im* format (FIREwork), the exact values contain in the array will be save in binary format as 'float32' data. Otherwise with the other format (png, jpg, etc.) the image will be normalize *(0, 1)* and convert to *uint8*, thus values are between *(0, 255)*. In this case you loosing the exact luminance of the original image.

**Examples**

::

	>>> im = range(128 * 128)
	>>> im = array(im, 'float32')
	>>> im = im.reshape((128, 128))
	>>> image_write(im, 'test.png')
	>>> image_write(im, 'test.im')
