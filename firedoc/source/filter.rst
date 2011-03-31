Filter
======

filter_2d_Gaussian
------------------

imf = **filter_2d_Gaussian** (im, sig)

*Aplly a 2D Gaussian filter to an image*

**Parameters**

``im`` 2D Numpy array as image

``sig`` Sigma value of the filter

**Returns**

``imf`` Image filtered as 2D Numpy array

filter_2d_Metz
--------------

imf = **filter_2d_Metz** (im, N, sig)

*Apply a 2D Metz filter to an image*

**Parameters**

``im`` 2D Numpy array as image

``N`` Power of the filter, if *N=0* is equivalent to a Gaussian filter.

``sig`` Sigma value of the filter

**Returns**

``imf`` Image filtered as 2D Numpy array


filter_2d_tanh_lp
-----------------

imf = **filter_2d_tanh_lp** (im, a, fc)

*Apply a 2D lowpass hyperbolic tangent filter to image*

**Parameters**

``im`` 2D Numpy array as image

``a`` Smooth factor of the slope

``fc`` Cut-off frequency (Nyquist frequency)

**Return**

``imf`` Image filtered as 2D Numpy array

**Notes**

See function filter_build_3d_tanh_lp for a plot of transfert function with different values of *a*.


filter_3d_Butterworth_lp
------------------------

volf = **filter_3d_Butterworth_lp** (vol, order, fc)

*Apply a 3D lowpass Butterworth filter to a volume*

**Parameters**

``vol`` Input volume to be filtered, 3D Numpy array ('float32')

``order`` Order of the filter

``fc`` Cutoff frequency

**Returns**

``volf`` 3D Numpy array after filtering

**Notes**

**Examples**

::

	>>> volf = filter_3d_Butterworth_lp(vol, 4, 0.2)

filter_3d_Gaussian
------------------

volf = **filter_3d_Gaussian** (vol, sig)

*Apply a 3D Gaussian filter to a volume*

**Parameters**

``vol`` Input volume to be filtered, 3D Numpy array ('float32')

``sig`` Standard deviation

**Returns**

``volf`` 3D Numpy array after filtering

**Notes**

**Examples**

::

	>>> volf = filter_3d_Gaussian(vol, 0.2)

filter_3d_Metz
--------------

volf = **filter_3d_Metz** (vol, N, sig)

*Apply a 3D Metz filter to a volume*

**Parameters**

``vol`` Input volume to be filtered, 3D Numpy array ('float32')

``N`` Power of the filter

``sig`` Standard deviation

**Returns**

``volf`` 3D Numpy array after filtering

**Notes**

**Examples**

::

	>>> volf = filter_3d_Metz(vol, 2, 0.1)

	
filter_3d_tanh_lp
-----------------

volf = **filter_3d_tanh_lp** (vol, a, fc)

*Apply a 3D lowpass hyperbolic tangent filter to a volume*

**Parameters**

``vol`` Input volume to be filtered, 3D Numpy array ('float32')

``a`` Smoothnest parameter

``fc`` Cutoff frequency

**Returns**

``volf`` 3D Numpy array after filtering

**Notes**

**Examples**

::

	>>> volf = filter_3d_tanh_lp(vol, 0.1, 0.2)

	
	
filter_build_1d_Butterworth_lp
------------------------------

Same function as filter_build_3d_Butterworth_lp, but in 1d

filter_build_1d_Gaussian
------------------------

Same function as filter_build_3d_Gaussian, but in 1d

filter_build_1d_Metz
--------------------

Same function as filter_build_3d_Metz, but in 1d

filter_build_1d_tanh_lp
-----------------------

Same function as filter_build_3d_tanh_lp, but in 1d

filter_build_2d_Butterworth_lp
------------------------------

Same function as filter_build_3d_Butterworth_lp, but in 2d

filter_build_2d_Gaussian
------------------------

Same function as filter_build_3d_Gaussian, but in 2d

filter_build_2d_Metz
--------------------

Same function as filter_build_3d_Metz, but in 2d

filter_build_2d_tanh_lp
-----------------------

Same function as filter_build_3d_tanh_lp, but in 2d

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

The transfert function is defined with a symmetry in order to be applied directly to the Fourrier space.

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

The transfert function is defined with a symmetry in order to be applied directly to the Fourrier space.

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

The transfert function is defined with a symmetry in order to be applied directly to the Fourrier space.

**Examples**

::

	>>> H = filter_build_3d_Metz(141, 2, 0.2)


filter_build_3d_tanh_lp
-----------------------

H = **filter_build_3d_tanh_lp** (size, a, fc)

*Build a 3D lowpass hyperbolic tangent filter*

**Parameters**

``size`` Size of the edge of the 3D transfert function.

``a`` Smooth factor of the slope.

``fc`` Cut-off frequency.

**Returns**

``H`` Numpy array of 3 dimensions *(size, size, size)*, which contains coefficients of the transfert function.

**Notes**

The transfert function is defined with a symmetry in order to be applied directly to the Fourrier space. In order to know which value of smoothness apply to your filter, refert the figure above where some values of *a* was plotted.

.. image:: data/tanl.png
   :scale: 50 %


**Examples**

::

	>>> H = filter_build_3d_tanh_lp(141, 0.1, 0.2)

	
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

filter_profil
-------------

p, f = **filter_profil** (H)

*Return the profil of any filter*

**Parameters**

``H`` Transfert function, can be 3D or 2D Numpy array

**Returns**

``p`` Value of profil

``f`` Nyquist frequency for each profil's values

