#!/usr/bin/env pythonfx
#
# This file is part of FIREwork
# 
# FIREwork is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIREwork is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
#
# FIREwork Copyright (C) 2008 - 2011 Julien Bert 

from OpenGL.GLUT       import *
from OpenGL.GL         import *
from OpenGL.GLU        import *

# Volume rendering by opengl
def render_gl_volume_surf(vol):
    from sys        import exit
    from numpy      import zeros, array
    from render_c   import render_gl_draw_pixels, render_volume_surf, render_image_color
    
    global phi, theta, scale
    global xmouse, ymouse, lmouse, rmouse, wmouse
    global w, h
    global lutr, lutg, lutb
    global jetr, jetb, jetg, hotr, hotg, hotb, hsvr, hsvb, hsvg, gray
    global color_flag

    wz, wy, wx  = vol.shape
    w, h        = 320, 240
    vol         = vol / vol.max()
    #vol        *= 255
    #vol         = vol.astype('uint8')
    color_flag       = 0
    lmouse, rmouse   = 0, 0
    xmouse, ymouse   = 0.0, 0.0
    wmouse           = 0.0
    phi, theta       = 0.0, 0.0
    scale            = 1.0
    mip              = zeros((h, w), 'float32')
    mapr             = zeros((h, w), 'float32')
    mapg             = zeros((h, w), 'float32')
    mapb             = zeros((h, w), 'float32')
    
    def build_map_color():
        global jetr, jetb, jetg, hotr, hotg, hotb, hsvr, hsvb, hsvg, gray
        global lutr, lutg, lutb
        
        jetr = zeros((256), 'float32')
        jetg = zeros((256), 'float32')
        jetb = zeros((256), 'float32')
        hotr = zeros((256), 'float32')
        hotg = zeros((256), 'float32')
        hotb = zeros((256), 'float32')
        hsvr = zeros((256), 'float32')
        hsvg = zeros((256), 'float32')
        hsvb = zeros((256), 'float32')
        gray = zeros((256), 'float32')

        # jet
        up  = array(range(0, 255,  3), 'float32')
        dw  = array(range(255, 0, -3), 'float32')
        stp = 85
        jetr[stp:2*stp]   = up
        jetr[2*stp:]      = 255
        jetg[0:stp]       = up
        jetg[stp:2*stp]   = 255
        jetg[2*stp:3*stp] = dw
        jetb[0:stp]       = 255
        jetb[stp:2*stp]   = dw
        jetr /= 255.0
        jetg /= 255.0
        jetb /= 255.0

        # hot
        up  = array(range(0, 255,  3), 'float32')
        stp = 85
        hotr[0:stp]       = up
        hotr[stp:]        = 255
        hotg[stp:2*stp]   = up
        hotg[2*stp:]      = 255
        hotb[2*stp:3*stp] = up
        hotb[3*stp:]      = 255
        hotr /= 255.0
        hotg /= 255.0
        hotb /= 255.0

        # hsv
        up  = array(range(0, 255,  5), 'float32')
        dw  = array(range(255, 0, -5), 'float32')
        stp = 51
        hsvr[0:stp]       = dw
        hsvr[3*stp:4*stp] = up
        hsvr[4*stp:]      = 255
        hsvg[0:2*stp]     = 255
        hsvg[2*stp:3*stp] = dw
        hsvb[stp:2*stp]   = up
        hsvb[2*stp:4*stp] = 255
        hsvb[4*stp:5*stp] = dw
        hsvr /= 255.0
        hsvg /= 255.0
        hsvb /= 255.0

        # gray
        gray = array(range(256), 'float32')
        gray /= 255.0

        # default
        lutr = gray.copy()
        lutg = gray.copy()
        lutb = gray.copy()

    def init():
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glShadeModel(GL_FLAT)

    def draw_HUD():
        global phi, theta, color_flag
        
        txt = 'Volume %ix%ix%i  phi %6.2f theta %6.2f' % (wz, wy, wx, phi, theta)
        glRasterPos2i(0, 1)
        for char in txt: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))

        if   color_flag == 0: txt2 = 'Gray color map'
        elif color_flag == 1: txt2 = 'Jet color map'
        elif color_flag == 2: txt2 = 'Hot color map'
        elif color_flag == 3: txt2 = 'HSV color map'
        glRasterPos2i(0, h-12)
        for char in txt2: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))


    def display():
        global phi, theta, scale, wmouse
        global lutr, lutg, lutb
        glClear (GL_COLOR_BUFFER_BIT)
        glRasterPos2i(0, 0)

        # get mip
        render_volume_surf(vol, mip, phi, theta, scale, wmouse)
        # color map
        render_image_color(mip, mapr, mapg, mapb, lutr, lutg, lutb)
        # draw
        render_gl_draw_pixels(mapr, mapg, mapb)
        # draw HUD
        draw_HUD()

        glutSwapBuffers()        
        
    def reshape(neww, newh):
        glViewport (0, 0, w, h)
        glMatrixMode (GL_PROJECTION)
        glLoadIdentity ()
        gluOrtho2D(0.0, w, 0.0, h)
        #glMatrixMode (GL_MODELVIEW)
        #glLoadIdentity()

    def keyboard(key, x, y):
        global jetr, jetb, jetg, hotr, hotg, hotb, hsvr, hsvb, hsvg, gray
        global lutr, lutg, lutb, color_flag, phi, theta
        if key == chr(27): sys.exit(0)
        elif key == 'a':   phi   += .5
        elif key == 'z':   phi   -= .5
        elif key == 'q':   theta += .5
        elif key == 's':   theta -= .5
        elif key == 'c':
            color_flag += 1
            if color_flag > 3: color_flag = 0
            if color_flag == 0:
                lutr = gray.copy()
                lutg = gray.copy()
                lutb = gray.copy()
            elif color_flag == 1:
                lutr = jetr.copy()
                lutg = jetg.copy()
                lutb = jetb.copy()
            elif color_flag == 2:
                lutr = hotr.copy()
                lutg = hotg.copy()
                lutb = hotb.copy()
            elif color_flag == 3:
                lutr = hsvr.copy()
                lutg = hsvg.copy()
                lutb = hsvb.copy()
                
        glutPostRedisplay()

    def mouse_click(button, state, x, y):
        global lmouse, rmouse, xmouse, ymouse, wmouse

        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN: lmouse = 1
            elif state == GLUT_UP:
                lmouse = 0
                xmouse = 0
                ymouse = 0

        if button == GLUT_RIGHT_BUTTON:
            if state == GLUT_DOWN: rmouse = 1
            elif state == GLUT_UP:
                rmouse = 0
                xmouse = 0
                ymosue = 0

        # GLUT_WHEEL_UP   = 3
        # GLUT_WHEEL_DOWN = 4
        if button == 3:
            wmouse += 0.005
            if wmouse >= wz: wmouse = wz-1

        if button == 4:
            wmouse -= 0.005
            if wmouse < 0: wmouse = 0

        glutPostRedisplay()
            
    def mouse_move(x, y):
        global xmouse, ymouse, lmouse, rmouse, phi, theta, scale
        if lmouse:
            if xmouse == 0 and ymouse == 0:
                xmouse = x
                ymouse = y
                return
            else:
                dx      = x - xmouse
                dy      = y - ymouse
                xmouse  = x
                ymouse  = y
                phi    += dx * 0.025
                theta  += dy * 0.025
                glutPostRedisplay()

        if rmouse:
            if xmouse == 0:
                xmouse = x
                return
            else:
                dx = x- xmouse
                xmouse = x
                scale += dx * 0.01
                glutPostRedisplay()
        
    glutInit(sys.argv)
    glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize (w, h)
    glutInitWindowPosition (100, 100)
    glutCreateWindow ('Viewer - FIREwork')
    init()
    build_map_color()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse_click)
    glutMotionFunc(mouse_move)
    glutMainLoop()

# MIP volume rendering by opengl
def render_gl_volume_mip(vol):
    from sys        import exit
    from numpy      import zeros, array
    from render_c   import render_gl_draw_pixels, render_volume_mip, render_image_color
    
    global phi, theta, scale
    global xmouse, ymouse, lmouse, rmouse
    global w, h
    global lutr, lutg, lutb
    global jetr, jetb, jetg, hotr, hotg, hotb, hsvr, hsvb, hsvg, petb, petr, petg, gray
    global color_flag

    wz, wy, wx  = vol.shape
    w, h        = 320, 240
    vol         = vol / vol.max()
    #vol        *= 255
    #vol         = vol.astype('uint8')
    color_flag       = 0
    lmouse, rmouse   = 0, 0
    xmouse, ymouse   = 0.0, 0.0
    phi, theta       = 0.0, 0.0
    scale            = 1.0
    mip              = zeros((h, w), 'float32')
    mapr             = zeros((h, w), 'float32')
    mapg             = zeros((h, w), 'float32')
    mapb             = zeros((h, w), 'float32')
    
    def build_map_color():
        global jetr, jetb, jetg, hotr, hotg, hotb, hsvr, hsvb, hsvg, petb, petr, petg, gray
        global lutr, lutg, lutb
        
        jetr = zeros((256), 'float32')
        jetg = zeros((256), 'float32')
        jetb = zeros((256), 'float32')
        hotr = zeros((256), 'float32')
        hotg = zeros((256), 'float32')
        hotb = zeros((256), 'float32')
        hsvr = zeros((256), 'float32')
        hsvg = zeros((256), 'float32')
        petb = zeros((256), 'float32')
        petr = zeros((256), 'float32')
        petg = zeros((256), 'float32')
        hsvb = zeros((256), 'float32')

        gray = zeros((256), 'float32')

        # jet
        up  = array(range(0, 255,  3), 'float32')
        dw  = array(range(255, 0, -3), 'float32')
        stp = 85
        jetr[stp:2*stp]   = up
        jetr[2*stp:]      = 255
        jetg[0:stp]       = up
        jetg[stp:2*stp]   = 255
        jetg[2*stp:3*stp] = dw
        jetb[0:stp]       = 255
        jetb[stp:2*stp]   = dw
        jetr /= 255.0
        jetg /= 255.0
        jetb /= 255.0

        # hot
        up  = array(range(0, 255,  3), 'float32')
        stp = 85
        hotr[0:stp]       = up
        hotr[stp:]        = 255
        hotg[stp:2*stp]   = up
        hotg[2*stp:]      = 255
        hotb[2*stp:3*stp] = up
        hotb[3*stp:]      = 255
        hotr /= 255.0
        hotg /= 255.0
        hotb /= 255.0

        # hsv
        up  = array(range(0, 255,  5), 'float32')
        dw  = array(range(255, 0, -5), 'float32')
        stp = 51
        hsvr[0:stp]       = dw
        hsvr[3*stp:4*stp] = up
        hsvr[4*stp:]      = 255
        hsvg[0:2*stp]     = 255
        hsvg[2*stp:3*stp] = dw
        hsvb[stp:2*stp]   = up
        hsvb[2*stp:4*stp] = 255
        hsvb[4*stp:5*stp] = dw
        hsvr /= 255.0
        hsvg /= 255.0
        hsvb /= 255.0

        # pet
        up2 = array(range(0, 255, 4), 'uint8') #  64
        up3 = array(range(0, 255, 8), 'uint8') #  32
        dw  = array(range(255, 0, -8), 'uint8') #  32
        petr[0:64]   = 0
        petg[0:64]   = 0
        petb[0:64]   = up2
        petr[64:128]   = up2
        petg[64:128]   = 0
        petb[64:128]   = 255
        petr[128:160] = 255
        petg[128:160] = 0
        petb[128:160] = dw
        petr[160:224] = 255
        petg[160:224] = up2
        petb[160:224] = 0
        petr[224:256] = 255
        petg[224:256] = 255
        petb[224:256] = up3
        petr /= 255.0
        petg /= 255.0
        petb /= 255.0

        # gray
        gray = array(range(256), 'float32')
        gray /= 255.0

        # default
        lutr = gray.copy()
        lutg = gray.copy()
        lutb = gray.copy()

    def init():
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glShadeModel(GL_FLAT)

    def draw_HUD():
        global phi, theta, color_flag
        
        txt = 'Volume %ix%ix%i  phi %6.2f theta %6.2f' % (wz, wy, wx, phi, theta)
        glRasterPos2i(0, 1)
        for char in txt: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))

        if   color_flag == 0: txt2 = 'Gray color map'
        elif color_flag == 1: txt2 = 'Jet color map'
        elif color_flag == 2: txt2 = 'Hot color map'
        elif color_flag == 3: txt2 = 'HSV color map'
        elif color_flag == 4: txt2 = 'PET color map'
        glRasterPos2i(0, h-12)
        for char in txt2: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))


    def display():
        global phi, theta, scale
        global lutr, lutg, lutb
        glClear (GL_COLOR_BUFFER_BIT)
        glRasterPos2i(0, 0)

        # get mip
        render_volume_mip(vol, mip, phi, theta, scale)
        # color map
        render_image_color(mip, mapr, mapg, mapb, lutr, lutg, lutb)
        # draw
        render_gl_draw_pixels(mapr, mapg, mapb)
        # draw HUD
        draw_HUD()

        glutSwapBuffers()        
        
    def reshape(neww, newh):
        glViewport (0, 0, w, h)
        glMatrixMode (GL_PROJECTION)
        glLoadIdentity ()
        gluOrtho2D(0.0, w, 0.0, h)
        #glMatrixMode (GL_MODELVIEW)
        #glLoadIdentity()

    def keyboard(key, x, y):
        global jetr, jetb, jetg, hotr, hotg, hotb, hsvr, hsvb, hsvg, gray
        global lutr, lutg, lutb, color_flag, phi, theta
        if key == chr(27): sys.exit(0)
        elif key == 'a':   phi   += .5
        elif key == 'z':   phi   -= .5
        elif key == 'q':   theta += .5
        elif key == 's':   theta -= .5
        elif key == 'c':
            color_flag += 1
            if color_flag > 4: color_flag = 0
            if color_flag == 0:
                lutr = gray.copy()
                lutg = gray.copy()
                lutb = gray.copy()
            elif color_flag == 1:
                lutr = jetr.copy()
                lutg = jetg.copy()
                lutb = jetb.copy()
            elif color_flag == 2:
                lutr = hotr.copy()
                lutg = hotg.copy()
                lutb = hotb.copy()
            elif color_flag == 3:
                lutr = hsvr.copy()
                lutg = hsvg.copy()
                lutb = hsvb.copy()
            elif color_flag == 4:
                lutr = petr.copy()
                lutg = petg.copy()
                lutb = petb.copy()
                
        glutPostRedisplay()

    def mouse_click(button, state, x, y):
        global lmouse, rmouse, xmouse, ymouse

        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN: lmouse = 1
            elif state == GLUT_UP:
                lmouse = 0
                xmouse = 0
                ymouse = 0

        if button == GLUT_RIGHT_BUTTON:
            if state == GLUT_DOWN: rmouse = 1
            elif state == GLUT_UP:
                rmouse = 0
                xmouse = 0
                ymosue = 0
            
    def mouse_move(x, y):
        global xmouse, ymouse, lmouse, rmouse, phi, theta, scale
        if lmouse:
            if xmouse == 0 and ymouse == 0:
                xmouse = x
                ymouse = y
                return
            else:
                dx      = x - xmouse
                dy      = y - ymouse
                xmouse  = x
                ymouse  = y
                phi    += dx * 0.025
                theta  += dy * 0.025
                glutPostRedisplay()

        if rmouse:
            if xmouse == 0:
                xmouse = x
                return
            else:
                dx = x- xmouse
                xmouse = x
                scale += dx * 0.01
                glutPostRedisplay()
        
    glutInit(sys.argv)
    glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize (w, h)
    glutInitWindowPosition (100, 100)
    glutCreateWindow ('Viewer - FIREwork')
    init()
    build_map_color()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse_click)
    glutMotionFunc(mouse_move)
    glutMainLoop()

# display slices of a 3D volume
def render_gl_volume_slices(vol):
    from sys        import exit
    from numpy      import zeros, array
    from render_c   import render_gl_draw_pixels, render_image_color
    
    global islice
    global w, h
    global lutr, lutg, lutb
    global jetr, jetb, jetg, hotr, hotg, hotb, hsvr, hsvb, hsvg, petr, petb, petg, gray
    global color_flag

    wz, wy, wx  = vol.shape
    w, h        = wx, wy
    #vol         = vol / vol.max()
    #vol        *= 255
    #vol         = vol.astype('uint8')
    color_flag       = 0
    lmouse, rmouse   = 0, 0
    xmouse, ymouse   = 0.0, 0.0
    islice           = 0
    imslice          = zeros((h, w), 'float32')
    mapr             = zeros((h, w), 'float32')
    mapg             = zeros((h, w), 'float32')
    mapb             = zeros((h, w), 'float32')
    
    def build_map_color():
        global jetr, jetb, jetg, hotr, hotg, hotb, hsvr, hsvb, hsvg, petr, petb, petg, gray
        global lutr, lutg, lutb
        
        jetr = zeros((256), 'float32')
        jetg = zeros((256), 'float32')
        jetb = zeros((256), 'float32')
        hotr = zeros((256), 'float32')
        hotg = zeros((256), 'float32')
        hotb = zeros((256), 'float32')
        hsvr = zeros((256), 'float32')
        hsvg = zeros((256), 'float32')
        hsvb = zeros((256), 'float32')
        petr = zeros((256), 'float32')
        petg = zeros((256), 'float32')
        petb = zeros((256), 'float32')
        gray = zeros((256), 'float32')

        # jet
        up  = array(range(0, 255,  3), 'float32')
        dw  = array(range(255, 0, -3), 'float32')
        stp = 85
        jetr[stp:2*stp]   = up
        jetr[2*stp:]      = 255
        jetg[0:stp]       = up
        jetg[stp:2*stp]   = 255
        jetg[2*stp:3*stp] = dw
        jetb[0:stp]       = 255
        jetb[stp:2*stp]   = dw
        jetr /= 255.0
        jetg /= 255.0
        jetb /= 255.0

        # hot
        up  = array(range(0, 255,  3), 'float32')
        stp = 85
        hotr[0:stp]       = up
        hotr[stp:]        = 255
        hotg[stp:2*stp]   = up
        hotg[2*stp:]      = 255
        hotb[2*stp:3*stp] = up
        hotb[3*stp:]      = 255
        hotr /= 255.0
        hotg /= 255.0
        hotb /= 255.0

        # hsv
        up  = array(range(0, 255,  5), 'float32')
        dw  = array(range(255, 0, -5), 'float32')
        stp = 51
        hsvr[0:stp]       = dw
        hsvr[3*stp:4*stp] = up
        hsvr[4*stp:]      = 255
        hsvg[0:2*stp]     = 255
        hsvg[2*stp:3*stp] = dw
        hsvb[stp:2*stp]   = up
        hsvb[2*stp:4*stp] = 255
        hsvb[4*stp:5*stp] = dw
        hsvr /= 255.0
        hsvg /= 255.0
        hsvb /= 255.0

        # pet
        up2 = array(range(0, 255, 4), 'uint8') #  64
        up3 = array(range(0, 255, 8), 'uint8') #  32
        dw  = array(range(255, 0, -8), 'uint8') #  32
        petr[0:64]   = 0
        petg[0:64]   = 0
        petb[0:64]   = up2
        petr[64:128]   = up2
        petg[64:128]   = 0
        petb[64:128]   = 255
        petr[128:160] = 255
        petg[128:160] = 0
        petb[128:160] = dw
        petr[160:224] = 255
        petg[160:224] = up2
        petb[160:224] = 0
        petr[224:256] = 255
        petg[224:256] = 255
        petb[224:256] = up3
        petr /= 255.0
        petg /= 255.0
        petb /= 255.0

        # gray
        gray = array(range(256), 'float32')
        gray /= 255.0

        # default
        lutr = gray.copy()
        lutg = gray.copy()
        lutb = gray.copy()

    def init():
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glShadeModel(GL_FLAT)

    def draw_HUD():
        global islice, color_flag
        
        txt = 'Volume %ix%ix%i  slice #%i' % (wz, wy, wx, islice)
        glRasterPos2i(0, 1)
        for char in txt: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))

        if   color_flag == 0: txt2 = 'Gray color map'
        elif color_flag == 1: txt2 = 'Jet color map'
        elif color_flag == 2: txt2 = 'Hot color map'
        elif color_flag == 3: txt2 = 'HSV color map'
        elif color_flag == 4: txt2 = 'PET color map'
        glRasterPos2i(0, h-12)
        for char in txt2: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))

    def display():
        global islice
        global lutr, lutg, lutb
        glClear (GL_COLOR_BUFFER_BIT)
        glRasterPos2i(0, 0)

        # get mip
        #render_volume_mip(vol, mip, phi, theta, scale)
        imslice = vol[int(islice), :, :]
        # color map
        render_image_color(imslice, mapr, mapg, mapb, lutr, lutg, lutb)
        # draw
        render_gl_draw_pixels(mapr, mapg, mapb)
        # draw HUD
        draw_HUD()

        glutSwapBuffers()        
        
    def reshape(neww, newh):
        glViewport (0, 0, w, h)
        glMatrixMode (GL_PROJECTION)
        glLoadIdentity ()
        gluOrtho2D(0.0, w, 0.0, h)
        #glMatrixMode (GL_MODELVIEW)
        #glLoadIdentity()

    def keyboard(key, x, y):
        global jetr, jetb, jetg, hotr, hotg, hotb, hsvr, hsvb, hsvg, petr, petg, petb, gray
        global lutr, lutg, lutb, color_flag
        if key == chr(27): sys.exit(0)
        elif key == 'c':
            color_flag += 1
            if color_flag > 4: color_flag = 0
            if color_flag == 0:
                lutr = gray.copy()
                lutg = gray.copy()
                lutb = gray.copy()
            elif color_flag == 1:
                lutr = jetr.copy()
                lutg = jetg.copy()
                lutb = jetb.copy()
            elif color_flag == 2:
                lutr = hotr.copy()
                lutg = hotg.copy()
                lutb = hotb.copy()
            elif color_flag == 3:
                lutr = hsvr.copy()
                lutg = hsvg.copy()
                lutb = hsvb.copy()
            elif color_flag == 4:
                lutr = petr.copy()
                lutg = petg.copy()
                lutb = petb.copy()
                
        glutPostRedisplay()

    def mouse_click(button, state, x, y):
        global islice
        # GLUT_WHEEL_UP   = 3
        # GLUT_WHEEL_DOWN = 4

        if button == 3:
            islice += 0.5
            if islice >= wz: islice = wz-1

        if button == 4:
            islice -= 0.5
            if islice < 0: islice = 0

        glutPostRedisplay()
        
    glutInit(sys.argv)
    glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize (w, h)
    glutInitWindowPosition (100, 100)
    glutCreateWindow ('Viewer - FIREwork')
    init()
    build_map_color()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse_click)
    glutMainLoop()

