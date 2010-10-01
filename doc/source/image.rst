<!-- Local IspellDict: american -->

Image
=====

image_atodB
-----------

imdb = **image_atodB** (im)

*Return the image in dB values*

**Parameters**

``im`` 2D Numpy array as image

**Returns**

``imdb`` 2D Numpy array as image

image_fft
---------

imf = **image_fft** (im)

*Compute the 2D FFT of image*

**Parameters**

``im`` 2D Numpy array as image, must be in 'float32' and square

**Returns**

``imf`` 2D Numpy array of FFT complexe values ('complex'). The spectrum is already shift.


image_frc
---------

fsc, freq = **image_frc** (im1, im2)

*Compute the Fourier Ring Correlation between two images*

**Parameters**

``im1, im2`` Two 2D Numpy array as images

**Returns**

``fsc`` 1D array of frc values

``freq`` 1D array of Nyquist frequencies for each frc values

**Notes**

Input images must be not normalize before computing the frc, in order to avoid (negative value).

image_ginput
------------

pts = **image_ginput** (im, n, [mapcolor])

*Get points position acquired by the mouse on a specify image*

**Parameters**

``im`` 2D Numpy array as image

``n`` Number of points required

``mapcolor`` Mapcolor to display the image, by default it is 'gray'

**Returns**

``pts`` List of points define as a 1D array pts = [x1, y1, xi, yi, ..., xn, yn]



image_ifft
----------

im = **image_ifft** (imf)

*Compute the inverse 2D FFT of image*

**Parameters**

``imf`` 2D Numpy array of FFT complexe values

**Returns**

``im`` The inverse FFT as 2D numpy array ('float32')


image_infos
-----------

**image_infos** (im)

*Display some usefull informations about an image (size, min, max, mean and std)*

**Parameters**

``im`` A 2D Numpy array as image


image_logscale
--------------

imlog = **image_logscale** (im)

*Return a image after streching the values in logscale*

**Parameters**

``im`` 2D Numpy array as image

**Returns**

``imlog`` 2D Numpy array as image

image_mask_circle
-----------------

mask = **image_mask_circle** (ny, nx, rad)

*Create a mask circle centred to the image*

**Parameters**

``ny, nx`` Size of the image which contains the mask

``rad`` Radius of the mask circle

**Returns**

``mask`` 2D Numpy array as image which contains the mask

image_mask_edge_square
----------------------

mask = **image_mask_edge_square** (ny, nx, c)

*Create a 2D mask compose of edge square of 1 pix thin*

**Parameters**

``ny, nx`` Size of the image

``c`` The half size of the larger square

**Returns**

``mask`` A 2D Numpy array

image_mask_square
-----------------

mask = **image_mask_square** (ny, nx, c)

*Create a 2D mask compose of square center to the image*

**Parameters**

``ny, nx`` Size of the image

``c`` The half size of the larger square

**Returns**

``mask`` A 2D Numpy array

image_mire_edge_square
----------------------

mire = **image_mire_edge_square** (ny, nx, step)

*Create a mire compose of a set of square edge*

**Parameters**

``ny, nx`` Size of the image

``step`` Step between each repeat edge

**Returns**

``mire`` A 2D Numpy array

**Notes**

Build base on image_mask_edge_square function

image_noise
-----------

noise = **image_noise** (ny, nx, sigma, [model])

*Build a 2D noise (white noise) according a model (Gaussian or Poisson)*

**Parameters**

``ny, nx`` Size of noise image

``sigma`` Sigma value, define the delta value around zeros i.e. how large is the distribution, and how strong is the noise.

``model`` Gaussian model or Poisson model (default is Gaussian)

**Returns**

``noise`` The noise image, 2D Numpy array ('float32')


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

image_periodogram
-----------------

per = **image_periodogram** (im)

*Return the periodogram of an image*

**Parameters**

``im`` A 2D Numpy array as image

**Returns**

``per`` A 2D Numpy array

**Notes**

Same as Power Spectrum (image_pows)
	
image_pows
----------

pows = **image_pows** (im)

*Return the power spectrum of an image*

**Parameters**

``im`` A 2D Numpy array as image

**Returns**

``pows`` A 2D Numpy array

image_ra
--------

val = **image_ra** (im)

*Compute the Radial Average of an image*

**Parameters**

``im`` A 2D Numpy array as image

**Returns**

``val`` 1D array, which contains the values of RA


image_raps
----------

val, freq = **image_raps** (im)

*Compute the Radial Averaging Power Spectrum from an image*

**Parameters**

``im`` A 2D Numpy array as image ('float32')

**Returns**

``val`` 1D array, which contains the values of the RAPS

``freq`` 1D array of Nyquist frequencies for each values of RAPS

**Notes**

The input image is not normalize i.e. the mean is not equal to zeros

image_show
----------

**image_show** (im, [mapcolor])

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


image_snr_from_zncc
-------------------

snr = **image_snr_from_zncc** (signal, noise)

*Compute the Signal-Noise-Ratio according the ZNCC coefficient between 2 images*

**Parameters**

``signal`` Image without noise as reference, 2D Numpy array ('float32')

``noise`` Image with noise, 2D Numpy array ('float32')

**Returns**

``snr`` Value of snr

image_stats_mask
----------------

min, max, mean, std = **image_stats_mask** (im, mask)

*Get statistic values on an image under a specified mask*

**Parameters**

``im`` 2D Numpy array as image.

``mask`` 2D Numpy array as mask (value must be 0.0 or 1.0)

**Return**

``min`` Minimum value under the mask

``max`` Maximum value under the mask

``mean`` Mean value under the mask

``std`` Standard deviation under the mask


image_stats_ROI_circle
----------------------

ROI, min, max, mean, std = **image_stats_ROI_circle** (im, cx, cy, rad)

*Get statistic values on an image only for a ROI with a circle shape*

**Parameters**

``im`` 2D Numpy array as image to be analysed.

``cx, cy`` Position of the circle ROI on the image in pixel

``rad`` Radius of the circle ROI

**Returns**

``ROI`` Image (2D Numpy array) with the ROI used

``min`` The min value on ROI

``max`` The max value on ROI

``mean`` The mean value on ROI

``std`` The standard deviation value on ROI

image_stitch
------------

newim = **image_stitch** (im1, im2)

*Stitch two images in one, i.e. put side by side two images, usefull to display in once*

**Parameters**

``im1, im2`` 2D Numpy array as images

**Returns**

``newim`` 2D Numpy array

**Examples**

::

	>>> im1 = image_open('im1.png')
	>>> im2 = image_open('im2.png')
	>>> image_show(image_stitch(im1, im2))

image_threshold_down
--------------------

newim = **image_threshold_down** (im, th, val)

*Replace values of an image if they are less or equal to a threshold value*

**Parameters**

``im`` 2D Numpy array as image

``th`` Threshold value

``val`` New replace value where the pixel value is lower than *th*

**Returns**

``newim`` New image as 2D Numpy array

image_threshold_up
------------------

newim = **image_threshold_up** (im, th, val)

*Replace values of an image if they are more or equal to a threshold value*

**Parameters**

``im`` 2D Numpy array as image

``th`` Threshold value

``val`` New replace value where the pixel value is more than *th*

**Returns**

``newim`` New image as 2D Numpy array


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

image_write_mapcolor
--------------------

**image_write_mapcolor** (im, filename, [colormap])

*Save a 2D Numpy array as an image with false color*

**Parameters**

``im`` Name of 2D Numpy array. The value format must be in *float32*.

``filename`` Name of the file you want to export the image. Different kind of format is supported like *bmp*, *png*, *tif* and *jpg*.

``colormap`` Specify the mapcolor of the false color transformation on the image, the default value is 'jet', but it can be 'hot', and 'hsv' as well.

**Examples**

::

	>>> im = range(128 * 128)
	>>> im = array(im, 'float32')
	>>> im = im.reshape((128, 128))
	>>> image_write_mapcolor(im, 'test.png', 'hot')

	
image_zncc
----------

ccc = **image_zncc** (im1, im2)

*Return the Zero-mean Normalized Cross Correlation Coefficient between 2 images*

**Parameters**

``im1, im2`` Two images, must be 2D Numpy array ('float32')

**Returns**

``ccc`` Value of ZNCC.
