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


image_show
----------

**image_show** (im)

*Display an image*

**Parameters**

``im`` Image name, must be a Numpy array of 2 dimensions.

**Returns**

**Notes**

viewer.py

**Examples**

::

	>>> im = image_open('test.png')
	>>> image_show(im)

