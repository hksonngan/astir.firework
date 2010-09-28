Misc
====

curve_smooth
------------

newval = **curve_smooth** (val, order)

*Smooth a value vector by calculating the mean of a sliding window*

**Parameters**

``val`` 1D Numpy array which contains values to be smoothed

``order`` Number of time the window is slided

**Returns**

``newval`` 1D Numpy array with values smoothed

**Notes**

The window size was set to two, so to increase the smoothness of the curve just increase the number of time that the window is slided.

plot
----

**plot** (x, y)

*Draw a plot with x and y*

**Parameters**

``x, y`` Two 1D Numpy array, with the values of x and y respectively (must be the same length size)



plot_filter_profil
------------------

**plot_filter_profil** (H)

*Plot the profile of any filter*

**Parameters**

``H`` Transfert function, can be 3D or 2D Numpy array

**Notes**

See function filter_profil

plot_frc
--------

**plot_frc** (im1, im2)

*Plot the Fourier Ring Correlation between two images*

**Parameters**

``im1, im2`` 2D Numpy array as images (sizes must be square)

**Notes**

See function image_frc

plot_raps
---------

**plot_raps** (im)

*Plot the Rotational Average Power Spectrum of an image*

**Parameters**

``im`` 2D Numpy array as image (sizes must be square)

**Notes**

See function image_raps


prefix_SI
---------

txt = **prefix_SI** (val)

*Create a text with engineer SI prefix format (k, M, G, T)*

**Parameters**

``val`` Value to be format.

**Returns**

``txt`` A text with SI format

**Examples**

::

	>>> print prefix_SI(12500)
	12.50 k

time_format
-----------

txt = **time_format** (t)

*Format a value in second to a nice h:m:s format*

**Parameters**

``t`` Time value in second.

**Returns**

``txt`` Text with h:m:s format

**Examples**

::

	>>> print time_format(12345.6)
	 03 h 25 m 45 s 600 ms
	>>> print time_format(123.45)
	 02 m 03 s 450 ms



wait
----

**wait** ()

*To get a break in your code :-). This function stop your script until you hit the return key.*


