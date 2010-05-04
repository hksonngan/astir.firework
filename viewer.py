#!/usr/bin/env python
from OpenGL.GLUT       import *
from OpenGL.GL         import *
from OpenGL.GLU        import *
from numpy             import array, arange, zeros, flipud
from sys               import exit
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

    '''
    color_func.SetColorSpaceToHSV();
    color_func.HSVWrapOn();
    color_func.AddHSVPoint( 0.0, 4.0/6.0, 1.0, 1.0);
    color_func.AddHSVPoint( vmax/4.0, 2.0/6.0, 1.0, 1.0);
    color_func.AddHSVPoint( vmax/2.0, 1.0/6.0, 1.0, 1.0);
    color_func.AddHSVPoint( vmax, 5.0/6.0, 1.0, 1.0);
    '''

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

def image_show(mat):
    import matplotlib.pyplot as plt
    import thread

    def fun(mat):
        h, w = mat.shape
        fig  = plt.figure()
        ax   = fig.add_subplot(111)
        cax  = ax.imshow(mat, interpolation='nearest', cmap='hot')
        ax.set_title('Viewer - FIREwork : %i x %i' % (w, h))
        min  = mat.min()
        max  = mat.max()
        cbar = fig.colorbar(cax, ticks=[min, max])
        cbar.ax.set_yticklabels(['%5.3f' % min, '%5.3f' % max])
        plt.show()

    thread.start_new_thread(fun, (mat,))
    
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
