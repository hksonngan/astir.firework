#include <GL/gl.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

void omp_vec_square(float* data, int n) {
	int i;
	#pragma omp parallel for shared(data) private(i)
	for(i=0; i<n; ++i) {data[i] = data[i] * data[i];}
}

void kernel_draw_voxels(int* posxyz, int npos, float* val, int nval, float gamma, float thres){
	int ind, n, x, y, z;
	float l;
	for (n=0; n<nval; ++n) {
		ind = 3 * n;
		x = posxyz[ind];
		y = posxyz[ind+1];
		z = posxyz[ind+2];
		l = val[n];
		if (l <= thres) {continue;}
		l *= gamma;
		glColor4f(1.0, 1.0, 1.0, l);
		// face 0
		glBegin(GL_QUADS);
		glNormal3f(-1, 0, 0);
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y+1.0, z); // 2
		glVertex3f(x, y+1.0, z+1.0); // 3
		glVertex3f(x, y, z+1.0); // 4
		glEnd();
		// face 1
		glBegin(GL_QUADS);
		glNormal3f(0, 1, 0);
		glVertex3f(x, y+1, z+1); // 3
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y+1, z+1); // 7
		glEnd();
		// face 2 
		glBegin(GL_QUADS);
		glNormal3f(1, 0, 0);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y, z+1); // 4
		glEnd();
		// face 3
		glBegin(GL_QUADS);
		glNormal3f(0, -1, 0);
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y, z+1); // 0
		glEnd();
		// face 4
		glBegin(GL_QUADS);
		glNormal3f(0, 0, 1);
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x, y, z); // 1
		glEnd();
		// face 5
		glBegin(GL_QUADS);
		glNormal3f(0, 0, -1);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x, y, z+1); // 0
		glVertex3f(x, y+1, z+1); // 3
		glEnd();
		
	}
	glColor4f(1.0, 1.0, 1.0, 1.0);
}

void kernel_draw_voxels_edge(int* posxyz, int npos, float* val, int nval, float thres){
	int ind, n, x, y, z;
	float l;
	for (n=0; n<nval; ++n) {
		ind = 3 * n;
		x = posxyz[ind];
		y = posxyz[ind+1];
		z = posxyz[ind+2];
		l = val[n];
		if (l <= thres) {continue;}
		// face 0
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(-1, 0, 0);
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y+1.0, z); // 2
		glVertex3f(x, y+1.0, z+1.0); // 3
		glVertex3f(x, y, z+1.0); // 4
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y+1.0, z); // 2
		glVertex3f(x, y+1.0, z+1.0); // 3
		glVertex3f(x, y, z+1.0); // 4
		glEnd();
		// face 1
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(0, 1, 0);
		glVertex3f(x, y+1, z+1); // 3
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y+1, z+1); // 7
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glNormal3f(0, 1, 0);
		glVertex3f(x, y+1, z+1); // 3
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y+1, z+1); // 7
		glEnd();
		// face 2
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(1, 0, 0);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y, z+1); // 4
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glNormal3f(1, 0, 0);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y, z+1); // 4
		glEnd();
		// face 3
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(0, -1, 0);
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y, z+1); // 0
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glNormal3f(0, -1, 0);
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x, y, z); // 1
		glVertex3f(x, y, z+1); // 0
		glEnd();
		// face 4
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(0, 0, 1);
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x, y, z); // 1
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glNormal3f(0, 0, 1);
		glVertex3f(x+1, y, z); // 5
		glVertex3f(x+1, y+1, z); // 6
		glVertex3f(x, y+1, z); // 2
		glVertex3f(x, y, z); // 1
		glEnd();
		// face 5
		glColor4f(1.0, 1.0, 1.0, l);
		glBegin(GL_QUADS);
		glNormal3f(0, 0, -1);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x, y, z+1); // 0
		glVertex3f(x, y+1, z+1); // 3
		glEnd();
		glColor3f(0.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		glNormal3f(0, 0, -1);
		glVertex3f(x+1, y+1, z+1); // 7
		glVertex3f(x+1, y, z+1); // 4
		glVertex3f(x, y, z+1); // 0
		glVertex3f(x, y+1, z+1); // 3
		glEnd();
	}
	glColor4f(1.0, 1.0, 1.0, 1.0);
}

// Draw a line in 2D space by Digital Differential Analyzer method (modified version to 1D)
void kernel_draw_2D_line_DDA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val) {
	int length, i;
	double x, y;
	double xinc, yinc;
	
	length = abs(x2 - x1);
	if (abs(y2 - y1) > length) {length = abs(y2 - y1);}
	xinc = (double)(x2 - x1) / (double) length;
	yinc = (double)(y2 - y1) / (double) length;
	x    = x1 + 0.5;
	y    = y1 + 0.5;
	for (i=0; i<=length; ++i) {
		mat[(int)y * wx + (int)x] += val;
		x = x + xinc;
		y = y + yinc;
	}
}

// Draw a line in 2D space by Bresenham's Line Algorithm (modified version 1D)
void kernel_draw_2D_line_BLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val) {
	int x, y;
	int dx, dy;
	int xinc, yinc;
	int balance;
	if (x2 >= x1) {
		dx = x2 - x1;
		xinc = 1;
	} else {
		dx = x1 - x2;
		xinc = -1;
	}
	if (y2 >= y1) {
		dy = y2 - y1;
		yinc = 1;
	} else {
		dy = y1 - y2;
		yinc = -1;
	}
	x = x1;
	y = y1;
	if (dx >= dy) {
		dy <<= 1;
		balance = dy - dx;
		dx <<= 1;
		while (x != x2) {
			mat[y * wx + x] += val;
			if (balance >= 0) {
				y = y + yinc;
				balance = balance - dx;
			}
			balance = balance + dy;
			x = x + xinc;
		}
		mat[y * wx + x] += val;
	} else {
		dx <<= 1;
		balance = dx - dy;
		dy <<=1;
		while (y != y2) {
			mat[y * wx + x] += val;
			if (balance >= 0) {
				x = x + xinc;
				balance = balance - dy;
			}
			balance = balance + dx;
			y = y + yinc;
		}
		mat[y * wx + x] += val;
	}
}

// Draw a line in 2D space by Wu's Line Algorithm (modified version 1D)
void kernel_draw_2D_line_WLA(float* mat, int wy, int wx, int x1, int y1, int x2, int y2, float val) {
	int dy = y2 - y1;
	int dx = x2 - x1;
	int stepx, stepy;

	if (dy < 0) { dy = -dy;  stepy = -1; } else { stepy = 1; }
	if (dx < 0) { dx = -dx;  stepx = -1; } else { stepx = 1; }

	mat[y1 * wx + x1] += val;
	mat[y2 * wx + x2] += val;
	
	if (dx > dy) {
		
		int length = (dx - 1) >> 2;
		int extras = (dx - 1) & 3;
		int incr2 = (dy << 2) - (dx << 1);
		if (incr2 < 0) {
			int c = dy << 1;
			int incr1 = c << 1;
			int d =  incr1 - dx;
			for (int i = 0; i < length; i++) {
				x1 = x1 + stepx;
				x2 = x2 - stepx;
				if (d < 0) {                    // Pattern:
					mat[y1 * wx + x1] += val;   //
					x1 = x1 + stepx;            // x o o
					mat[y1 * wx + x1] += val;
					mat[y2 * wx + x2] += val;
					x2 = x2 - stepx;
					mat[y2 * wx * x2] += val;
					d += incr1;
				} else {
					if (d < c) {                                 // Pattern:
						mat[y1 * wx + x1] += val;                //       o
						x1 = x1 + stepx;                         //   x o
						y1 = y1 + stepy;
						mat[y1 * wx * x1] += val;
						mat[y2 * wx * x2] += val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
					} else {
						y1 = y1 + stepy;                      // Pattern
						mat[y1 + wx + x1] += val;             //    o o
						x1 = x1 + stepx;                      //  x
						mat[y1 * wx + x1] += val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
						x1 = x1 - stepx;
						mat[y2 * wx * x2] += val;
					}
					d += incr2;
				}
			}
			if (extras > 0) {
				if (d < 0) {
					x1 = x1 + stepx;
					mat[y1 * wx * x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
					}
				} else 
                if (d < c) {
					x1 = x1 + stepx;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
					}
				} else {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
                        if (extras > 1) {
							x1 = x1 + stepx;
							mat[y1 * wx + x1] += val;
						}
                        if (extras > 2) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						}
                }
			}
		} else {
		    int c = (dy - dx) << 1;
			int incr1 = c << 1;
			int d =  incr1 + dx;
			for (int i = 0; i < length; i++) {
				x1 = x1 + stepx;
				x2 = x2 - stepx;
				if (d > 0) {
					y1 = y1 + stepy;           // Pattern
					mat[y1 * wx + x1] += val;  //      o
					x1 = x1 + stepx;           //    o
					y1 = y1 + stepy;           //   x
					mat[y1 * wx + x1] += val;
					y2 = y2 - stepy;
					mat[y2 * wx + x2] += val;
					x2 = x2 - stepx;
					y2 = y2 - stepy;
					mat[y2 * wx + x2] += val;
					d += incr1;
				} else {
					if (d < c) {
						mat[y1 * wx + x1] += val;  // Pattern
						x1 = x1 + stepx;           //      o
						y1 = y1 + stepy;           //  x o
						mat[y1 * wx + x1] += val;
						mat[y2 * wx + x2] += val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					} else {
						y1 = y1 + stepy;          // Pattern
						mat[y1 * wx + x1] += val; //    o  o
						x1 = x1 + stepx;          //  x
						mat[y1 * wx + x1] += val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
						x2 = x2- stepx;
						mat[y2 * wx + x2] += val;
					}
					d += incr2;
				}
			}
			if (extras > 0) {
				if (d > 0) {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
				} else 
                if (d < c) {
					x1 = x1 + stepx;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
					}
				} else {
					x1 = x1 - stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						if (d > c) {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						} else {
							x2 = x2 - stepx;
							mat[y2 * wx + x2] += val;
						}
					}
				}
            }
		}
    } else {
	    int length = (dy - 1) >> 2;
		int extras = (dy - 1) & 3;
		int incr2 = (dx << 2) - (dy << 1);
		if (incr2 < 0) {
			int c = dx << 1;
			int incr1 = c << 1;
			int d =  incr1 - dy;
			for (int i = 0; i < length; i++) {
				y1 = y1 + stepy;
				y2 = y2 - stepy;
				if (d < 0) {
					mat[y1 * wx + x1] += val;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					mat[y2 * wx + x2] += val;
					y2 = y2 - stepy;
					mat[y2 * wx + x2] += val;
					d += incr1;
				} else {
					if (d < c) {
						mat[y1 * wx + x1] += val;
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						mat[y2 * wx + x2] += val;
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					} else {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
					d += incr2;
				}
			}
			if (extras > 0) {
				if (d < 0) {
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
				} else 
                if (d < c) {
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
                } else {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
                }
			}
        } else {
			int c = (dx - dy) << 1;
			int incr1 = c << 1;
			int d =  incr1 + dy;
			for (int i = 0; i < length; i++) {
				y1 = y1 + stepy;
				y2 = y2 - stepy;
				if (d > 0) {
					x1 = x1 + stepx;
					mat[y1 * wx + x1] += val;
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					x2 = x2 - stepx;
					mat[y2 * wx + x2] += val;
					x2 = x2 - stepx;
					y2 = y2 - stepy;
					mat[y2 * wx + x2] += val;
					d += incr1;
				} else {
					if (d < c) {
						mat[y1 * wx + x1] += val;
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						mat[y2 * wx + x2] += val; 
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					} else {
						x1 = x1 + stepx;
						mat[y1 * wx + x1] += val;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
						x2 = x2 - stepx;
						mat[y2 * wx + x2] += val;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
					d += incr2;
				}
			}
			if (extras > 0) {
				if (d > 0) {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						x2 = x2 - stepx;
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
				} else
                if (d < c) {
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						x1 = x1 + stepx;
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
                    if (extras > 2) {
						y2 = y2 - stepy;
						mat[y2 * wx + x2] += val;
					}
				} else {
					x1 = x1 + stepx;
					y1 = y1 + stepy;
					mat[y1 * wx + x1] += val;
					if (extras > 1) {
						y1 = y1 + stepy;
						mat[y1 * wx + x1] += val;
					}
					if (extras > 2) {
						if (d > c)  {
							x2 = x2 - stepx;
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						} else {
							y2 = y2 - stepy;
							mat[y2 * wx + x2] += val;
						}
					}
				}
            }
        }
	}
}
