#!/usr/bin/env python
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
# FIREwork Copyright (C) 2008 - 2010 Julien Bert 

# display an image
def image_show(mat, map='gray'):
    import matplotlib.pyplot as plt

    h, w = mat.shape
    fig  = plt.figure()
    ax   = fig.add_subplot(111)
    cax  = ax.imshow(mat, interpolation='nearest', cmap=map)
    ax.set_title('Viewer - FIREwork : %i x %i' % (w, h))
    min  = mat.min()
    max  = mat.max()
    d    = (max - min) / 9.0
    lti  = [i*d+min for i in xrange(10)]
    txt  = ['%5.3f' % lti[i] for i in xrange(10)]
    cbar = fig.colorbar(cax, ticks=lti)
    cbar.ax.set_yticklabels(txt)
    plt.show()

# Get input values from an image
def image_ginput(im, n, map='gray'):
    from numpy import array
    import matplotlib.pyplot as plt
    #plt.figure(figsize=(1.4, 1.4))
    plt.figure(0)
    plt.imshow(im, interpolation='nearest', cmap=map)
    plt.colorbar()
    pts = plt.ginput(n)
    plt.show()
    plt.close(0)
    
    return array(pts, 'float32')
    
# plot 1D hitogram based on 1D data
def hist1D_plot(data, nbins):
    import matplotlib.pyplot as plt
    import matplotlib.mlab   as mlab

    n, bins, patches = plt.hist(data, nbins, facecolor='green', alpha=0.75)
    #print n
    #print bins
    #plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    plt.title('Viewer - FIREwork hist1D')
    #plt.axis([min(data), max(data), 0, max(n)])
    plt.grid(True)
    
    plt.show()

from OpenGL.GLUT       import *
from OpenGL.GL         import *
from OpenGL.GLU        import *

# volume rendering by opengl
def volume_show(vol):
    from numpy             import array, arange, zeros, flipud, take, sqrt
    from sys               import exit
    from kernel            import kernel_draw_voxels, kernel_draw_voxels_edge
    from utils             import volume_pack_cube

    global rotx, roty, rotz, scale
    global xmouse, ymouse, lmouse, rmouse
    global w, h
    global vec, lmap, lmapl, lmapc, lthr
    global flag_trans, flag_edge, flag_color
    global gamma, thres

    wz, wy, wx = vol.shape
    vol        = vol / vol.max()
    if not wx == wy == wz:
        # must be put in a cube
        vol = volume_pack_cube(vol)
        wz, wy, wx = vol.shape
    cz, cy, cx       = wz//2, wy//2, wx//2
    w, h             = 320, 240
    scale            = 3.0
    lmouse, rmouse   = 0, 0
    xmouse, ymouse   = 0.0, 0.0
    rotx, roty, rotz = 0.0, 0.0, 0.0
    vec, lmap        = [], []
    flag_trans       = 0
    flag_edge        = 0
    flag_color       = 0
    gamma            = 1.0
    thres            = 0.0

    def init():
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glShadeModel(GL_FLAT) # not gouraud (only cube)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.5, 0.5, 0.5, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glEnable(GL_COLOR_MATERIAL)

    def build_voxel():
        global vec, lmap, lmapl, lmapc, lthr
        buf1 = []
        buf2 = []
        for z in xrange(wz):
            for y in xrange(wy):
                for x in xrange(wx):
                    val = vol[z, y, x]
                    if val != 0:
                        buf1.extend([x, y, z])
                        buf2.append(val)
                        
        vec       = array(buf1, 'i')
        lthr      = array(buf2, 'f')
        lmapi     = lthr  * 255
        lthr      = sqrt(lthr)
        N         = len(lthr)
        lmapi     = lmapi.astype('int32')
        lutr      = zeros((256), 'int32')
        lutg      = zeros((256), 'int32')
        lutb      = zeros((256), 'int32')
        # jet color
        up        = array(range(0, 255,  3), 'int32')
        dw        = array(range(255, 0, -3), 'int32')
        stp       = 85
        lutr[stp:2*stp]   = up
        lutr[2*stp:]      = 255
        lutg[0:stp]       = up
        lutg[stp:2*stp]   = 255
        lutg[2*stp:3*stp] = dw
        lutb[0:stp]       = 255
        lutb[stp:2*stp]   = dw
        matr = take(lutr, lmapi)
        matg = take(lutg, lmapi)
        matb = take(lutb, lmapi)
        lmapc = zeros((3*N), 'float32')
        lmapl = zeros((3*N), 'float32')
        for i in xrange(N):
            ind = 3*i
            lmapc[ind]   = matr[i] / 255.0
            lmapc[ind+1] = matg[i] / 255.0
            lmapc[ind+2] = matb[i] / 255.0
            lmapl[ind]   = lthr[i] #**0.5 # increase brightness
            lmapl[ind+1] = lthr[i] #**0.5
            lmapl[ind+2] = lthr[i] #**0.5
        lmap = lmapl.copy()
        del matr, matg, matb, lutr, lutg, lutb, lmapi
        
    def draw_workspace():
        # draw workspace
        glBegin(GL_LINES)
        # front face
        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(0.0, 0.0, wz)
        glVertex3f(wx, 0.0, wz)
        glVertex3f(wx, 0.0, wz)
        glVertex3f(wx, wy, wz)
        glVertex3f(wx, wy, wz)
        glVertex3f(0.0, wy, wz)
        glVertex3f(0.0, wy, wz)
        glVertex3f(0.0, 0.0, wz)
        # back face
        glColor3f(1.0, 0.0, 0.0) # x axe
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(wx, 0.0, 0.0)
        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(wx, 0.0, 0.0)
        glVertex3f(wx, wy, 0.0)
        glVertex3f(wx, wy, 0.0)
        glVertex3f(0.0, wy, 0.0)
        glColor3f(0.0, 1.0, 0.0) # y axe
        glVertex3f(0.0, wy, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        # four remain edges
        glColor3f(0.0, 0.0, 1.0) # z axe
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, wz)
        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(wx, 0.0, 0.0)
        glVertex3f(wx, 0.0, wz)
        glVertex3f(0.0, wy, 0.0)
        glVertex3f(0.0, wy, wz)
        glVertex3f(wx, wy, 0.0)
        glVertex3f(wx, wy, wz)
        glEnd()

    def draw_HUD():
        global w, h, rotx, roty, rotz, scale
        txt = 'Volume %ix%ix%i rot x y z %6.2f %6.2f %6.2f scale %5.2f gamma %5.2f thr %5.2f' % (wx, wy, wz, rotx, roty, rotz, scale, gamma, thres)
        glRasterPos2i(-w//2, -h//2+1)
        for char in txt: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))

        txt2 = 't: transparency     e: edges     c: colors     7/1: +/- gamma    9/3: +/- threshold'
        glRasterPos2i(-w//2, h//2-12)
        for char in txt2: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))


    def display():
        global w, h, vec, lmap, flag_trans, flag_edge, gamma, lthr, thres
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # some options
        if flag_trans:
            glEnable(GL_BLEND)
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_LIGHTING)
            glDisable(GL_LIGHT0)
            #glDisable(GL_CULL_FACE)
            glDepthMask(GL_FALSE);
        else:
            glDisable(GL_BLEND)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            #glEnable(GL_CULL_FACE)
        
        glPushMatrix()
        glRotatef(rotx, 1.0, 0.0, 0.0)
        glRotatef(roty, 0.0, 1.0, 0.0)
        glRotatef(rotz, 0.0, 0.0, 1.0)
        glScalef(scale, scale, scale)
        glTranslatef(-cx, -cy, -cz)

        draw_workspace()
        if flag_edge: kernel_draw_voxels_edge(vec, lmap, lthr, thres)
        else:         kernel_draw_voxels(vec, lmap, lthr, gamma, thres)
        if flag_trans:
            glDepthMask(GL_TRUE)
        
        glTranslate(cx, cy, cz)
        glRotatef(-rotx, 1.0, 0.0, 0.0)
        glRotatef(-roty, 0.0, 1.0, 0.0)
        glRotatef(-rotz, 0.0, 0.0, 1.0)
        glPopMatrix()
         
        # draw HUD
        draw_HUD()
        glutSwapBuffers()        
        
    def reshape(neww, newh):
        # must be even, more easy for the next...
        neww = neww + neww % 2
        newh = newh + newh % 2
                
        glViewport (0, 0, neww, newh)
        glMatrixMode (GL_PROJECTION)
        glLoadIdentity ()
        glOrtho(-neww//2, neww//2, -newh//2, newh//2, -1000.0, 1000.0)
        glMatrixMode (GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -3*wz)
        w, h = neww, newh

    def keyboard(key, x, y):
        global rotx, roty, rotz, flag_trans, flag_edge, flag_color, gamma, thres, lmap, lmpal, lmapc
        if key == chr(27): sys.exit(0)
        elif key == 'a':   rotx += .5
        elif key == 'z':   rotx -= .5
        elif key == 'q':   roty += .5
        elif key == 's':   roty -= .5
        elif key == 'w':   rotz += .5
        elif key == 'x':   rotz -= .5
        elif key == 't':
            if flag_trans == 0:
                flag_trans = 1
                flag_edge  = 0
            else:               flag_trans = 0
        elif key == 'e':
            if flag_edge == 0:
                flag_edge  = 1
                flag_trans = 0
            else:               flag_edge = 0
        elif key == '7':
            gamma += 0.01
            if gamma >= 1.0: gamma = 1.0
        elif key == '1':
            gamma -= 0.01
            if gamma < 0.0: gamma = 0.0
        elif key == '9':
            thres += 0.01
            if thres >= 1.0: thres = 1.0
        elif key == '3':
            thres -= 0.01
            if thres < 0.0: thres = 0.0
        elif key == 'c':
            if flag_color:
                lmap = lmapl.copy()
                flag_color = 0
            else:
                lmap = lmapc.copy()
                flag_color = 1
        elif key == '1': print key

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
        global xmouse, ymouse, lmouse, rmouse, rotx, roty, scale
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
                roty   += dx * 0.25
                rotx   += dy * 0.25
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
    glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize (w, h)
    glutInitWindowPosition (100, 100)
    glutCreateWindow ('Viewer - FIREwork')
    init()
    build_voxel()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse_click)
    glutMotionFunc(mouse_move)
    glutMainLoop()



# MIP volume rendering by opengl
def volume_show_mip(vol):
    from sys    import exit
    from numpy  import zeros, array
    from kernel import kernel_draw_pixels, kernel_mip_volume_rendering, kernel_color_image
    
    global phi, theta, scale
    global xmouse, ymouse, lmouse, rmouse
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
        global phi, theta, scale
        global lutr, lutg, lutb
        glClear (GL_COLOR_BUFFER_BIT)
        glRasterPos2i(0, 0)

        # get mip
        kernel_mip_volume_rendering(vol, mip, phi, theta, scale)
        # color map
        kernel_color_image(mip, mapr, mapg, mapb, lutr, lutg, lutb)
        # draw
        kernel_draw_pixels(mapr, mapg, mapb)
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
