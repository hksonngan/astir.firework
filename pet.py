#!/usr/bin/env python

# create a list-mode from a simple simulate data
# detector
  # # #
# o o o #
# o x o #
# o o o #
  # # #
def pet2D_square_test_LOR(nx, ny, posx, posy, nbp):
    from numpy        import zeros, array
    from random       import random
    from math         import pi, cos, sin
    from numpy.random import poisson

    crystals = zeros((nx*ny, nx*ny), 'f')
    
    for p in xrange(nbp):
        # Poisson distribution
        if random() >= 0.5: x0 = posx + poisson(lam=1, size=(1))[0]
        else:               x0 = posx - poisson(lam=1, size=(1))[0]
        if random() >= 0.5: y0 = posy + poisson(lam=1, size=(1))[0]
        else:               y0 = posy - poisson(lam=1, size=(1))[0]

        # uniform angle
        alpha = random() * pi

        # photon back-to-back simulation
        g1_x, g2_x   = x0, x0
        g1_y, g2_y   = y0, y0
        id1, id2     = -1, -1
        if alpha >= (pi / 4.0) and alpha <= (3 * pi / 4.0):
            incx = cos(alpha)
            incy = 1
            while 1:
                g1_x += incx
                g1_y -= incy
                if g1_x <= 0:
                    id1 = 2 * nx + 2 * ny - int(g1_y) - 1
                    break
                if g1_x >= nx:
                    id1 = nx + int(g1_y) + 1
                    break
                if g1_y <= 0:
                    id1 = int(g1_x) + 1
                    break
            while 1:
                g2_x -= incx
                g2_y += incy
                if g2_x >= nx:
                    id2 = nx + int(g2_y) + 1
                    break
                if g2_x <= 0:
                    id2 = 2 * nx + 2 * ny - int(g2_y) + 1
                    break
                if g2_y >= ny:
                    id2 = 2 * nx + ny - int(g2_x) + 1
                    break
        else:
            if alpha >= (3 * pi / 4.0): incx = -1
            else:                       incx =  1
            incy = sin(alpha)
            while 1:
                g1_x += incx
                g1_y -= incy
                if g1_x <= 0:
                    id1 = 2 * nx + 2 * ny - int(g1_y) + 1
                    break
                if g1_x >= nx:
                    id1 = nx + int(g1_y) + 1
                    break
                if g1_y <= 0:
                    id1 = int(g1_x) + 1
                    break
            while 1:
                g2_x -= incx
                g2_y += incy
                if g2_x >= nx:
                    id2 = nx + int(g2_y) + 1
                    break
                if g2_x <= 0:
                    id2 = 2 * nx + 2 * ny - int(g2_y) + 1
                    break
                if g2_y >= ny:
                    id2 = 2 * nx + ny - int(g2_x) + 1
                    break

        crystals[id2, id1] += 1.0

    # build list-mode
    lm = []
    for id2 in xrange(ny*nx):
        for id1 in xrange(ny*nx):
            val = int(crystals[id2, id1])
            if val != 0: lm.extend([id1, id2, val])
    lm = array(lm, 'uint8')

    return lm

def pet2D_square_backproj_LOR(lor, nx, ny):
    N = len(lor)
    for n in xrange(lor):
        id1, id2, val = lor[n]
        #face1 = id1 // 
        
            
