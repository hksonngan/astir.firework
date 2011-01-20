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

from OpenGL.GLUT       import *
from OpenGL.GL         import *
from OpenGL.GLU        import *
from numpy             import array, arange, zeros, flipud
from sys               import exit


'''
import vtk
# volume rendering by vtk
def viewer_volume_vtk(vol):
    wz, wy, wx = vol.shape
    vmax       = vol.max()
    data_importer = vtk.vtkImageImport()
    data_string   = vol.tostring()
    data_importer.CopyImportVoidPointer(data_string, len(data_string))
    data_importer.SetDataScalarTypeToUnsignedShort()
    data_importer.SetNumberOfScalarComponents(1)
    data_importer.SetDataExtent(0, wx-1, 0, wy-1, 0, wz-1)
    data_importer.SetWholeExtent(0, wx-1, 0, wy-1, 0, wz-1)
    data_importer.SetDataSpacing(1.0, 1.0, 1.0)

    alpha_func    = vtk.vtkPiecewiseFunction()
    alpha_func.AddPoint(0, 0.0)
    #alpha_func.AddPoint(60, 0.0)
    alpha_func.AddPoint(vmax, 1.0)
    
    
    color_func    = vtk.vtkColorTransferFunction()
    color_func.AddRGBPoint(vmax, 1.0, 1.0, 1.0)

    #color_func.SetColorSpaceToHSV();
    #color_func.HSVWrapOn();
    #color_func.AddHSVPoint( 0.0, 4.0/6.0, 1.0, 1.0);
    #color_func.AddHSVPoint( vmax/4.0, 2.0/6.0, 1.0, 1.0);
    #color_func.AddHSVPoint( vmax/2.0, 1.0/6.0, 1.0, 1.0);
    #color_func.AddHSVPoint( vmax, 5.0/6.0, 1.0, 1.0);

    vol_property  = vtk.vtkVolumeProperty()
    vol_property.SetColor(color_func)
    vol_property.SetScalarOpacity(alpha_func)
    vol_property.ShadeOff()
    #vol_property.SetInterpolationTypeToLinear()
    
    composite_func = vtk.vtkVolumeRayCastCompositeFunction()
    vol_mapper     = vtk.vtkVolumeRayCastMapper()
    vol_mapper.SetVolumeRayCastFunction(composite_func)
    vol_mapper.SetInputConnection(data_importer.GetOutputPort())
    #vol_mapper = vtk.vtkVolumeTextureMapper2D()
    #vol_mapper.SetInputConnection(data_importer.GetOutputPort())
    
    volume = vtk.vtkVolume()
    volume.SetMapper(vol_mapper)
    volume.SetProperty(vol_property)

    renderer  = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)

    #cube = vtk.vtkCubeSource()
    #cubemapper = vtk.vtkPolyDataMapper()
    #cubemapper.SetInput(cube.GetOutput())
    #cubeactor = vtk.vtkActor()
    #cubeactor.SetMapper(cubemapper)
    #renderer.AddActor(cubeactor)
    
    renderer.AddVolume(volume)
    renderer.SetBackground(0, 0, 0)
    renderWin.SetSize(400, 400)

    def exitCheck(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)

    renderWin.AddObserver('AbortCheckEvent', exitCheck)

    renderInteractor.Initialize()
    renderWin.Render()
    renderInteractor.Start()
'''
   
# volume rendering by opengl
def volume_show(vol):
    from kernel import kernel_draw_voxels, kernel_draw_voxels_edge
    global rotx, roty, rotz, scale
    global xmouse, ymouse, lmouse, rmouse
    global w, h
    global vec, lmap
    global flag_trans, flag_edge
    global gamma, thres

    wz, wy, wx       = vol.shape
    cz, cy, cx       = wz//2, wy//2, wx//2
    w, h             = 800, 500
    scale            = 3.0
    lmouse, rmouse   = 0, 0
    xmouse, ymouse   = 0.0, 0.0
    rotx, roty, rotz = 0.0, 0.0, 0.0
    vec, lmap        = [], []
    flag_trans       = 0
    flag_edge        = 0
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
        global vec, lmap
        buf1 = []
        buf2 = []
        for z in xrange(wz):
            for y in xrange(wy):
                for x in xrange(wx):
                    val = vol[z, y, x]
                    if val != 0:
                        buf1.extend([x, y, z])
                        buf2.append(val)
                        
        vec  = array(buf1, 'i')
        lmap = array(buf2, 'f')

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
        txt = 'Volume %ix%ix%i rot x y z = %6.2f %6.2f %6.2f scale = %5.2f' % (wx, wy, wz, rotx, roty, rotz, scale)
        glRasterPos2i(-w//2, -h//2)
        for char in txt: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))    

    def display():
        global w, h, vec, lmap, flag_trans, flag_edge, gamma, thres
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
        if flag_edge: kernel_draw_voxels_edge(vec, lmap, thres)
        else:         kernel_draw_voxels(vec, lmap, gamma, thres)
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
        global rotx, roty, rotz, flag_trans, flag_edge, gamma, thres
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
                thres      = 0.0
                flag_edge  = 0
            else:               flag_trans = 0
        elif key == 'e':
            if flag_edge == 0:
                flag_edge  = 1
                flag_trans = 0
            else:               flag_edge = 0
        elif key == '+':
            if flag_trans:  gamma += 0.01
            else:           thres -= 0.01
        elif key == '-':
            if flag_trans:
                gamma -= 0.01
                if gamma < 0: gamma = 0.0
            else:             thres += 0.01

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

# NOT USE: some test to improve volume rendering by opengl
def viewer_volume_2(vol):
    global rotx, roty, rotz
    global xmouse, ymouse
    global lmouse, rmouse
    global scale
    global w, h
    global wx, wy, wz
    global points, indices
    points, indices = 0, 0
    wz, wy, wx = vol.shape
    cz, cy, cx = wz//2, wy//2, wx//2
    
    w, h             = 800, 500
    scale            = 3.0
    lmouse, rmouse   = 0, 0
    xmouse, ymouse   = 0.0, 0.0
    rotx, roty, rotz = 0.0, 0.0, 0.0
    texture          = 0
    
    # init geometrically a cube
    normals = array([[-1.0, 0.0, 0.0], [0.0,  1.0,  0.0],
                     [ 1.0, 0.0, 0.0], [0.0, -1.0,  0.0],
                     [ 0.0, 0.0, 1.0], [0.0,  0.0, -1.0]])
    faces =   array([[0, 1, 2, 3], [3, 2, 6, 7],
                     [7, 6, 5, 4], [4, 5, 1, 0],
                     [5, 6, 2, 1], [7, 4, 0, 3]])
    vertexs = zeros((8, 3), 'f')                     
    vertexs[0][0] = vertexs[1][0] = vertexs[2][0] = vertexs[3][0] =  0
    vertexs[4][0] = vertexs[5][0] = vertexs[6][0] = vertexs[7][0] =  wx
    vertexs[0][1] = vertexs[1][1] = vertexs[4][1] = vertexs[5][1] =  0
    vertexs[2][1] = vertexs[3][1] = vertexs[6][1] = vertexs[7][1] =  wx
    vertexs[0][2] = vertexs[3][2] = vertexs[4][2] = vertexs[7][2] =  wx
    vertexs[1][2] = vertexs[2][2] = vertexs[5][2] = vertexs[6][2] =  0
    texels  = zeros((8, 3), 'f')
    texels[0][0] = texels[1][0] = texels[2][0] = texels[3][0] =  0
    texels[4][0] = texels[5][0] = texels[6][0] = texels[7][0] =  1
    texels[0][1] = texels[1][1] = texels[4][1] = texels[5][1] =  0
    texels[2][1] = texels[3][1] = texels[6][1] = texels[7][1] =  1
    texels[0][2] = texels[3][2] = texels[4][2] = texels[7][2] =  1
    texels[1][2] = texels[2][2] = texels[5][2] = texels[6][2] =  0

    
    def init(vol):
        glClearColor (0.0, 0.0, 0.0, 0.0)
        #glEnable(GL_LIGHTING)
        #glEnable(GL_LIGHT0)
        #glShadeModel(GL_FLAT) # not gouraud (only cube)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        #glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.5, 0.5, 0.5, 1.0])
        #glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 1.0, 1.0, 1.0])
        #glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        #glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_3D)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)        
        #glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glDisable(GL_CULL_FACE)

        # Create Texture
        glGenTextures(1, texture)
        glBindTexture(GL_TEXTURE_2D, texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        
        #vol = vol.reshape((wx*wy*wz))
        glTexImage3D(GL_TEXTURE_3D, 0, GL_ALPHA, wx, wy, wz, 0, GL_ALPHA, GL_FLOAT, vol)

    def draw_cube():
        glEnable(GL_TEXTURE_3D)
        glBindTexture(GL_TEXTURE_3D, texture)
        glDepthMask(GL_FALSE)
        #glutSolidTeapot(16)
        for i in xrange(6):
            glBegin(GL_QUADS)
            glNormal3fv(normals[i])
            glTexCoord3fv(texels[faces[i][0]])
            glVertex3fv(vertexs[faces[i][0]])
            glTexCoord3fv(texels[faces[i][1]])
            glVertex3fv(vertexs[faces[i][1]])
            glTexCoord3fv(texels[faces[i][2]])
            glVertex3fv(vertexs[faces[i][2]])
            glTexCoord3fv(texels[faces[i][3]])
            glVertex3fv(vertexs[faces[i][3]])
            glEnd()
        glDisable(GL_TEXTURE_3D)
        glDepthMask(GL_TRUE)
        #glColor3f(1.0, 1.0, 1.0)
        #glutSolidCube(32)

    def draw_volume():
        #shift = zeros((3), 'f')
        #glDepthMask(GL_FALSE)
        #shift[0], shift[1], shift[2] = 0, wy-y, 0
        glEnable(GL_TEXTURE_3D)
        glBindTexture(GL_TEXTURE_3D, texture)

        glDepthMask(GL_TRUE)
            
        glColor4f(1.0, 1.0, 1.0, 1.0)

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

    # not use yet
    def draw_toolsbar():
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_QUADS)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(wtile, 0.0, 0.0)
        glVertex3f(wtile, -wtile, 0.0)
        glVertex3f(0.0, -wtile, 0.0)
        glEnd()
    
    def draw_HUD():
		global w, h, wx, wy, wz, rotx, roty, rotz, scale
		txt = 'Volume %ix%ix%i rot x y z = %6.2f %6.2f %6.2f scale = %5.2f' % (wx, wy, wz, rotx, roty, rotz, scale)
		glRasterPos2i(-w//2, -h//2)
		for char in txt: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))    

    def display():
        global w, h
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
      

        glPushMatrix()
        glRotatef(rotx, 1.0, 0.0, 0.0)
        glRotatef(roty, 0.0, 1.0, 0.0)
        glRotatef(rotz, 0.0, 0.0, 1.0)
        glScalef(scale, scale, scale)
        glTranslatef(-cx, -cy, -cz)
        draw_workspace()
        #draw_volume()
        draw_cube()
        glTranslate(cx, cy, cz)
        glRotatef(-rotx, 1.0, 0.0, 0.0)
        glRotatef(-roty, 0.0, 1.0, 0.0)
        glRotatef(-rotz, 0.0, 0.0, 1.0)
        glPopMatrix()
        
        # draw tools bar 
        #glTranslate(-w//2, 0.0, 0.0)
        #draw_toolsbar()
        #glTranslate(w//2, 0.0, 0.0)
        
        # draw HUD
        draw_HUD()
        glutSwapBuffers()        
        #glFlush ()

    def reshape (neww, newh):
        global w, h
        # must be even, more easy for the next...
        neww = neww + neww % 2
        newh = newh + newh % 2
        w, h = neww, newh
        
        glViewport (0, 0, w, h)
        glMatrixMode (GL_PROJECTION)
        glLoadIdentity ()
        #gluPerspective(65.0, 1.0, 1.0, 1000.0)
        glOrtho(-w//2, w//2, -h//2, h//2, -1000.0, 1000.0)
        #glFrustum (-1.0, 1.0, -1.0, 1.0, 1.5, 20.0)
        glMatrixMode (GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -3*wz)

    def keyboard(key, x, y):
        global rotx, roty, rotz
        if key == chr(27): sys.exit(0)
        elif key == 'a':   rotx += .5
        elif key == 'z':   rotx -= .5
        elif key == 'q':   roty += .5
        elif key == 's':   roty -= .5
        elif key == 'w':   rotz += .5
        elif key == 'x':   rotz -= .5
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
    init(vol)
    #build_quads(vol)
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse_click)
    glutMotionFunc(mouse_move)
    glutMainLoop()

# create a list-mode from a simple simulate data with only one point in space
def pet2D_square_test_1point_LOR(nx, posx, posy, nbp, rnd = 10):
    from kernel       import kernel_pet2D_square_gen_sim_ID
    from numpy        import zeros, array
    from numpy.random import poisson
    from random       import seed, random, randrange
    from math         import pi
    seed(10)

    crystals = zeros((nx*nx, nx*nx), 'f')
    image    = zeros((nx, nx), 'f')
    pp1      = poisson(lam=2.0, size=(nbp)).astype('f')
    ps1      = [randrange(-1, 2) for i in xrange(nbp)]
    pp2      = poisson(lam=2.0, size=(nbp)).astype('f')
    ps2      = [randrange(-1, 2) for i in xrange(nbp)]
    alpha    = [random()*pi for i in xrange(nbp)]
    res      = zeros((2), 'i')
    for p in xrange(nbp):
        x = posx + (ps1[p] * pp1[p])
        y = posy + (ps2[p] * pp2[p])
        kernel_pet2D_square_gen_sim_ID(res, x, y, alpha[p], nx)
        id1, id2 = res
        crystals[id2, id1] += 1.0
        image[y, x]  += 1.0

    # build LOR
    LOR_val = []
    LOR_id1 = []
    LOR_id2 = []
    for id2 in xrange(nx*nx):
        for id1 in xrange(nx*nx):
            val = int(crystals[id2, id1])
            if val != 0:
                LOR_val.append(val)
                LOR_id1.append(id1)
                LOR_id2.append(id2)

    LOR_val = array(LOR_val, 'i')
    LOR_id1 = array(LOR_id1, 'i')
    LOR_id2 = array(LOR_id2, 'i')

    return LOR_val, LOR_id1, LOR_id2, image

# create a list-mode from a simple simulate data with circle phantom
# image size is fixed to 65x65 with three differents activities
def pet2D_square_test_circle_LOR(nbp, rnd = 10):
    from kernel       import kernel_pet2D_square_gen_sim_ID
    from numpy        import zeros, array
    from numpy.random import poisson
    from numpy.random import seed as seed2
    from random       import seed, random, randrange
    from math         import pi
    seed(rnd)
    seed2(rnd)

    nx        = 65
    crystals  = zeros((nx*nx, nx*nx), 'f')
    image     = zeros((nx, nx), 'f')
    source    = []
    
    # three differents circle
    cx0, cy0, r0 = 32, 32, 16
    cx1, cy1, r1 = 36, 36, 7
    cx2, cy2, r2 = 26, 26, 2
    r02          = r0*r0
    r12          = r1*r1
    r22          = r2*r2
    for y in xrange(nx):
        for x in xrange(nx):
            if ((cx0-x)*(cx0-x) + (cy0-y)*(cy0-y)) <= r02:
                # inside the first circle
                if ((cx1-x)*(cx1-x) + (cy1-y)*(cy1-y)) <= r12:
                    # inside the second circle (do nothing)
                    continue
                
                if ((cx2-x)*(cx2-x) + (cy2-y)*(cy2-y)) <= r22:
                    # inside the third circle
                    source.extend([x, y, 5])
                    #image[y, x] = 5
                else:
                    # inside the first circle
                    source.extend([x, y, 1])
                    #image[y, x] = 1
                    
    nbpix  = len(source) // 3
    pp1    = poisson(lam=1.0, size=(nbp)).astype('f')
    ps1    = [randrange(-1, 2) for i in xrange(nbp)]
    pp2    = poisson(lam=1.0, size=(nbp)).astype('f')
    ps2    = [randrange(-1, 2) for i in xrange(nbp)]
    alpha  = [random()*pi for i in xrange(nbp)]
    ind    = [randrange(nbpix) for i in xrange(nbp)]
    res    = zeros((2), 'i')
    for p in xrange(nbp):
        x   = source[3*ind[p]]   + (ps1[p] * pp1[p])
        y   = source[3*ind[p]+1] + (ps2[p] * pp2[p])
        val = source[3*ind[p]+2]
        kernel_pet2D_square_gen_sim_ID(res, x, y, alpha[p], nx)
        id1, id2 = res
        crystals[id2, id1] += val
        image[y, x]  += source[3*ind[p]+2]

    # build LOR
    LOR_val = []
    LOR_id1 = []
    LOR_id2 = []
    for id2 in xrange(nx*nx):
        for id1 in xrange(nx*nx):
            val = int(crystals[id2, id1])
            if val != 0:
                LOR_val.append(val)
                LOR_id1.append(id1)
                LOR_id2.append(id2)

    LOR_val = array(LOR_val, 'i')
    LOR_id1 = array(LOR_id1, 'i')
    LOR_id2 = array(LOR_id2, 'i')

    return LOR_val, LOR_id1, LOR_id2, image



# build the sensibility matrix based in the system response matrix for all possible LOR
def pet2D_square_build_SM(nx):
    from numpy  import zeros, ones
    from kernel import kernel_build_2D_SRM_BLA
    from utils  import image_1D_projection

    nlor    = 6 * nx * nx  # pet 4 heads
    SRM     = zeros((nlor, nx * nx), 'float32')
    line    = zeros((4 * nlor), 'i')
    LOR_val = ones((nlor), 'i')

    # first head
    i = 0
    for x1 in xrange(nx):
        for y2 in xrange(nx):
            line[i:i+4] = [x1, 0, nx-1, y2]
            i += 4
        for x2 in xrange(nx):
            line[i:i+4] = [x1, 0, x2, nx-1]
            i += 4
        for y2 in xrange(nx):
            line[i:i+4] = [x1, 0, 0, y2]
            i += 4
    # second head
    for y1 in xrange(nx):
        for x2 in xrange(nx):
            line[i:i+4] = [nx-1, y1, x2, nx-1]
            i += 4
        for y2 in xrange(nx):
            line[i:i+4] = [nx-1, y1, 0, y2]
            i += 4
    # third head
    for x1 in xrange(nx):
        for y2 in xrange(nx):
            line[i:i+4] = [x1, nx-1, 0, y2]
            i += 4

    kernel_build_2D_SRM_BLA(SRM, LOR_val, line, nx)
    norm = image_1D_projection(SRM, 'x')
    SRM  = SRM.astype('f')
    for i in xrange(nlor): SRM[i] /= float(norm[i])
    
    return image_1D_projection(SRM, 'y')

# build the System Response Matrix for a list of LOR detected
def pet2D_square_build_SRM_LOR(LOR_val, LOR_id1, LOR_id2, nx):
    from numpy import zeros
    from kernel import kernel_build_2D_SRM_BLA
    
    nlor = len(LOR_val)
    SRM  = zeros((nlor, nx*nx), 'float32')
    N    = len(LOR_val)
    line = zeros((4 * N), 'i') # format [x1, y1, x2, y2, ct]

    # transform LOR index in x, y image space according the detector
    for n in xrange(N):
        id1  = LOR_id1[n]
        id2  = LOR_id2[n]
        val  = LOR_val[n]
        face = id1 // nx
        res  = id1 % nx
        if   face == 0:
            y1 = 0
            x1 = res
        elif face == 1:
            x1 = nx - 1
            y1 = res
        elif face == 2:
            y1 = nx - 1
            x1 = nx - res - 1
        elif face == 3:
            x1 = 0
            y1 = nx - res - 1

        face = id2 // nx
        res  = id2 % nx
        if   face == 0:
            y2 = 0
            x2 = res
        elif face == 1:
            x2 = nx - 1
            y2 = res
        elif face == 2:
            y2 = nx - 1
            x2 = nx - res - 1
        elif face == 3:
            x2 = 0
            y2 = nx - res - 1

        line[4*n:4*(n+1)] = [x1, y1, x2, y2]

    kernel_build_2D_SRM_BLA(SRM, LOR_val, line, nx)
            
    return SRM

# PET 2D ring scan create LOR event from a simple simulate phantom (three activities)
def pet2D_ring_simu_circle_phantom(nbcrystals, nbparticules, rnd = 10, mode='bin'):
    from numpy        import zeros, array
    from numpy.random import poisson
    from numpy.random import seed as seed2
    from random       import seed, random, randrange
    from math         import pi, sqrt, cos, sin
    from kernel       import kernel_pet2D_ring_gen_sim_ID, kernel_draw_2D_line_BLA
    seed(rnd)
    seed2(rnd)

    radius = int(nbcrystals / 2.0 / pi + 0.5)      # radius PET
    dia    = 2 * radius + 1                        # dia PET must be odd
    cxo    = cyo = radius                          # center PET
    if mode == 'bin': crystals = zeros((nbcrystals, nbcrystals), 'float32')
    image    = zeros((dia, dia), 'float32')
    source   = []

    # three differents circle
    cx0, cy0, r0 = cxo,    cyo,    16
    cx1, cy1, r1 = cx0+4,  cy0+4,   7
    cx2, cy2, r2 = cx0-6,  cy0-6,   2
    r02          = r0*r0
    r12          = r1*r1
    r22          = r2*r2
    for y in xrange(dia):
        for x in xrange(dia):
            if ((cx0-x)*(cx0-x) + (cy0-y)*(cy0-y)) <= r02:
                # inside the first circle
                if ((cx1-x)*(cx1-x) + (cy1-y)*(cy1-y)) <= r12:
                    # inside the second circle (do nothing)
                    continue
                if ((cx2-x)*(cx2-x) + (cy2-y)*(cy2-y)) <= r22:
                    # inside the third circle
                    source.extend([x, y, 5])
                    #image[y, x] = 5
                else:
                    # inside the first circle
                    source.extend([x, y, 1])
                    #image[y, x] = 1
                    
    nbpix  = len(source) // 3
    pp1    = poisson(lam=1.0, size=(nbparticules)).astype('int32')
    ps1    = [randrange(-1, 2) for i in xrange(nbparticules)]
    pp2    = poisson(lam=1.0, size=(nbparticules)).astype('int32')
    ps2    = [randrange(-1, 2) for i in xrange(nbparticules)]
    alpha  = [random()*pi for i in xrange(nbparticules)]
    ind    = [randrange(nbpix) for i in xrange(nbparticules)]
    res    = zeros((2), 'int32')
    lines  = zeros((4), 'int32')
    LOR_id1 = []
    LOR_id2 = []

    for p in xrange(nbparticules):
        x   = int(source[3*ind[p]]   + (ps1[p] * pp1[p]))
        y   = int(source[3*ind[p]+1] + (ps2[p] * pp2[p]))
        val = source[3*ind[p]+2]
        kernel_pet2D_ring_gen_sim_ID(res, x, y, alpha[p], radius)
        id1, id2 = res
        if id1 == nbcrystals: id1 = 0
        if id2 == nbcrystals: id2 = 0
        if mode == 'bin':
            crystals[id2, id1] += val
        else:
            # list-mode
            for n in xrange(val):
                LOR_id1.append(id1)
                LOR_id2.append(id2)
        image[y, x] += source[3*ind[p]+2]

    if mode == 'bin':
        # build LOR
        LOR_val = []
        for id2 in xrange(nbcrystals):
            for id1 in xrange(nbcrystals):
                val = int(crystals[id2, id1])
                if val != 0:
                    LOR_val.append(val)
                    LOR_id1.append(id1)
                    LOR_id2.append(id2)

        LOR_val = array(LOR_val, 'int32')
        LOR_id1 = array(LOR_id1, 'int32')
        LOR_id2 = array(LOR_id2, 'int32')

        return LOR_val, LOR_id1, LOR_id2, image
    else:
        # list-mode
        LOR_id1 = array(LOR_id1, 'int32')
        LOR_id2 = array(LOR_id2, 'int32')

        return None, LOR_id1, LOR_id2, image


# PET 2D Simulated ring scan build SRM by limiting the mem size and using sparse COO matrix
def pet2D_ring_simu_build_SRM_COO(lm_id1, lm_id2, chunk_size, nb_crystals, npix):
    from numpy  import zeros, append
    from kernel import kernel_pet2D_ring_LM_SRM_BLA, kernel_matrix_mat2coo
    from time import time
    
    totevents = len(lm_id1)
    ntime     = (totevents + chunk_size - 1) / chunk_size
    for itime in xrange(ntime):
        i_start   = int(round(float(totevents) / ntime * itime))
        i_stop    = int(round(float(totevents) / ntime * (itime+1)))
        nevents   = i_stop - i_start
        local_SRM = zeros((nevents, npix), 'float32')
        t1 = time()
        n_nonzeros = kernel_pet2D_ring_LM_SRM_BLA(local_SRM, lm_id1[i_start:i_stop], lm_id2[i_start:i_stop], nb_crystals)
        print '   raytracer', time() - t1, 's'
        # convert to COO sparse matrix
        t1 = time()
        local_vals = zeros((n_nonzeros), 'float32')
        local_rows = zeros((n_nonzeros), 'int32')
        local_cols = zeros((n_nonzeros), 'int32')
        print '   alloc mem', time() - t1, 's'
        t1 = time()
        kernel_matrix_mat2coo(local_SRM, local_vals, local_rows, local_cols, i_start, 0)
        print '   convert', time() - t1, 's'
        t1 = time()
        del local_SRM
        if itime != 0:
            vals = append(vals, local_vals)
            del local_vals
            rows = append(rows, local_rows)
            del local_rows
            cols = append(cols, local_cols)
            del local_cols
        else:
            vals = local_vals.copy()
            del local_vals
            rows = local_rows.copy()
            del local_rows
            cols = local_cols.copy()
            del local_cols
        print '   upmem', time() - t1, 's'

    return vals, rows, cols
        

# Open list-mode pre-compute data set (int format), values are entry-exit point of SRM matrix
def listmode_open_xyz_int(basename):
    from numpy import fromfile
    
    f  = open(basename + '.x1', 'rb')
    x1 = fromfile(file=f, dtype='uint16')
    x1 = x1.astype('uint16')
    f.close()

    f  = open(basename + '.y1', 'rb')
    y1 = fromfile(file=f, dtype='uint16')
    y1 = y1.astype('uint16')
    f.close()
    
    f  = open(basename + '.z1', 'rb')
    z1 = fromfile(file=f, dtype='uint16')
    z1 = z1.astype('uint16')
    f.close()
    
    f  = open(basename + '.x2', 'rb')
    x2 = fromfile(file=f, dtype='uint16')
    x2 = x2.astype('uint16')
    f.close()
    
    f  = open(basename + '.y2', 'rb')
    y2 = fromfile(file=f, dtype='uint16')
    y2 = y2.astype('uint16')
    f.close()
    
    f  = open(basename + '.z2', 'rb')
    z2 = fromfile(file=f, dtype='uint16')
    z2 = z2.astype('uint16')
    f.close()

    return x1, y1, z1, x2, y2, z2

# Open list-mode pre-compute data set (float format), values are entry-exit point of SRM matrix
def listmode_open_xyz_float(basename):
    from numpy import fromfile
    
    f  = open(basename + '.x1', 'rb')
    x1 = fromfile(file=f, dtype='float32')
    f.close()

    f  = open(basename + '.y1', 'rb')
    y1 = fromfile(file=f, dtype='float32')
    f.close()
    
    f  = open(basename + '.z1', 'rb')
    z1 = fromfile(file=f, dtype='float32')
    f.close()
    
    f  = open(basename + '.x2', 'rb')
    x2 = fromfile(file=f, dtype='float32')
    f.close()
    
    f  = open(basename + '.y2', 'rb')
    y2 = fromfile(file=f, dtype='float32')
    f.close()
    
    f  = open(basename + '.z2', 'rb')
    z2 = fromfile(file=f, dtype='float32')
    f.close()

    return x1, y1, z1, x2, y2, z2


# Open list-mode subset
def listmode_open_subset(filename, N_start, N_stop):
    from numpy import zeros
    f      = open(filename, 'r')
    nlines = N_stop - N_start
    lm_id1 = zeros((nlines), 'int32')
    lm_id2 = zeros((nlines), 'int32')
    for n in xrange(N_start):
        buf = f.readline()
    for n in xrange(nlines):
        id1, id2 = f.readline().split()
        lm_id1[n] = int(id1)
        lm_id2[n] = int(id2)
    f.close()

    return lm_id1, lm_id2

# Nb events ti list-mode file
def listmode_nb_events(filename):
    f = open(filename, 'r')
    n = 0
    while 1:
        buf = f.readline()
        if buf == '': break
        n += 1
        
    return n

# Open Sensibility Matrix
def listmode_open_SM(filename):
    from numpy import fromfile

    '''
    data = open(filename, 'r').readlines()
    Ns   = len(data)
    SM   = zeros((Ns), 'float32')
    for n in xrange(Ns):
        SM[n] = float(data[n])
    del data
    '''
    f  = open(filename, 'rb')
    SM = fromfile(file=f, dtype='int32')
    SM = SM.astype('float32')

    return SM

# PET 2D ring scan create LOR event from a simple simulate phantom (three activities) ONLY FOR ALLEGRO SCANNER
def listmode_simu_circle_phantom(nbparticules, ROIsize = 141, radius_detector = 432, respix = 4.0, rnd = 10):
    from numpy        import zeros, array
    from numpy.random import poisson
    from numpy.random import seed as seed2
    from random       import seed, random, randrange
    from math         import pi, sqrt, cos, sin, asin, acos
    seed(rnd)
    seed2(rnd)

    # Cst Allegro
    Ldetec    = 97.0 / float(respix)  # detector width (mm)
    Lcryst    = 4.3  / float(respix) # crystal width (mm)
    radius    = radius_detector / float(respix)
    border    = int((2*radius - ROIsize) / 2.0)
    cxo = cyo = ROIsize // 2
    image     = zeros((ROIsize, ROIsize), 'float32')
    source    = []

    # Generate an activity map with three differents circles
    cx0, cy0, r0 = cxo,    cyo,    16
    cx1, cy1, r1 = cx0+4,  cy0+4,   7
    cx2, cy2, r2 = cx0-6,  cy0-6,   2
    r02          = r0*r0
    r12          = r1*r1
    r22          = r2*r2
    for y in xrange(ROIsize):
        for x in xrange(ROIsize):
            if ((cx0-x)*(cx0-x) + (cy0-y)*(cy0-y)) <= r02:
                # inside the first circle
                if ((cx1-x)*(cx1-x) + (cy1-y)*(cy1-y)) <= r12:
                    # inside the second circle (do nothing)
                    continue
                if ((cx2-x)*(cx2-x) + (cy2-y)*(cy2-y)) <= r22:
                    # inside the third circle
                    source.extend([x, y, 5])
                    #image[y, x] = 5
                else:
                    # inside the first circle
                    source.extend([x, y, 1])
                    #image[y, x] = 1
                    
    nbpix  = len(source) // 3
    pp1    = poisson(lam=1.0, size=(nbparticules)).astype('int32')
    ps1    = [randrange(-1, 2) for i in xrange(nbparticules)]
    pp2    = poisson(lam=1.0, size=(nbparticules)).astype('int32')
    ps2    = [randrange(-1, 2) for i in xrange(nbparticules)]
    alpha  = [random()*2*pi for i in xrange(nbparticules)]
    ind    = [randrange(nbpix) for i in xrange(nbparticules)]
    
    IDD1 = zeros((nbparticules), 'int32')
    IDC1 = zeros((nbparticules), 'int32')
    IDD2 = zeros((nbparticules), 'int32')
    IDC2 = zeros((nbparticules), 'int32')
    p    = 0
    while p < nbparticules:
        x     = int(source[3*ind[p]]   + (ps1[p] * pp1[p]))
        y     = int(source[3*ind[p]+1] + (ps2[p] * pp2[p]))
        val   = source[3*ind[p]+2]
        image[y, x] += (source[3*ind[p]+2] * 10.0)
        x    += border
        y    += border
        a     = alpha[p]
        # compute line intersection with a circle (detectors)
	dx    = cos(a)
	dy    = sin(a)
	b     = 2 * (dx*(x-radius) + dy*(y-radius))
	c     = 2*radius*radius + x*x + y*y - 2*(radius*x + radius*y) - radius*radius
	d     = b*b - 4*c
	k0    = (-b + sqrt(d)) / 2.0
	k1    = (-b - sqrt(d)) / 2.0
	x1    = x + k0*dx
	y1    = y + k0*dy
	x2    = x + k1*dx
	y2    = y + k1*dy
	# compute angle from center of each point
	dx = x1 - radius #- border
	dy = y1 - radius #- border
	if abs(dx) > abs(dy):
            phi1 = asin(dy / float(radius))
            if phi1 < 0: # asin return -pi/2 < . < pi/2
                if dx < 0: phi1 = pi - phi1
                else:      phi1 = 2*pi + phi1
            else:
                if dx < 0: phi1 = pi - phi1 # mirror according y axe
	else:
            phi1 = acos(dx / float(radius))
            if dy < 0:     phi1 = 2*pi - phi1 # mirror according x axe
 	dx = x2 - radius #- border
	dy = y2 - radius #- border
	if abs(dx) > abs(dy):
            phi2 = asin(dy / float(radius))
            if phi2 < 0: # asin return -pi/2 < . < pi/2
                if dx < 0: phi2 = pi - phi2
                else:      phi2 = 2*pi + phi2
            else:
                if dx < 0: phi2 = pi - phi2 # mirror according y axe
	else:
            phi2 = acos(dx / float(radius))
            if dy < 0:     phi2 = 2*pi - phi2 # mirror according x axe

        # convert arc distance to ID detector and crystal
        phi1  = (2 * pi - phi1 + pi / 2.0) % (2 * pi)
        phi2  = (2 * pi - phi2 + pi / 2.0) % (2 * pi)
        pos1  = phi1 * radius
        pos2  = phi2 * radius
        idd1  = int(pos1 / Ldetec)
        pos1 -= (idd1 * Ldetec)
        idc1  = int(pos1 / Lcryst)
        idd2  = int(pos2 / Ldetec)
        pos2 -= (idd2 * Ldetec)
        idc2  = int(pos2 / Lcryst)

        for n in xrange(val):
            if p >= nbparticules: break
            IDD1[p] = idd1
            IDC1[p] = idc1
            IDD2[p] = idd2
            IDC2[p] = idc2
            p += 1
        if p >= nbparticules: break
        
    return image, IDC1, IDD1, IDC2, IDD2

# build the sensibility Matrix to 2D ring PET scan according all possibles LORs
def pet2D_ring_build_SM(nbcrystals):
    from numpy  import zeros
    from math   import cos, sin, pi, sqrt
    from kernel import kernel_pet2D_ring_build_SM
    from utils  import image_1D_projection
    radius = int(nbcrystals / 2.0 / pi + 0.5)      # radius PET
    dia    = 2 * radius + 1                        # dia PET must be odd
    cxo    = cyo = radius                          # center PET
    Nlor   = (nbcrystals-1) * (nbcrystals-1) / 2   # nb all possible LOR
    radius = float(radius)

    # build SRM for only the square image inside the ring of the PET
    SM = zeros((dia * dia), 'float32')
    for i in xrange(nbcrystals-1):
        nlor  = nbcrystals-(i+1)
        index = 0
        SRM   = zeros((nlor, dia * dia), 'float32')
        for j in xrange(i+1, nbcrystals):
            alpha1 = i / radius
            alpha2 = j / radius
            x1     = int(cxo + radius * cos(alpha1) + 0.5)
            x2     = int(cxo + radius * cos(alpha2) + 0.5)
            y1     = int(cyo + radius * sin(alpha1) + 0.5)
            y2     = int(cyo + radius * sin(alpha2) + 0.5)
            kernel_pet2D_ring_build_SM(SRM, x1, y1, x2, y2, dia, index)
            index += 1
            
        # sum by step in order to decrease the memory for this stage
        norm = image_1D_projection(SRM, 'x')
        for n in xrange(nlor): SRM[n] /= float(norm[n])
        res = image_1D_projection(SRM, 'y')
        SM += res
 
    return SM


# volume rendering by opengl 2011-01-20 12:09:21
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

