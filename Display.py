import pygame, OpenGL
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

class Display:
    def __init__(s, fr_height, fr_width, scl, filename):
        s.img = pygame.image.load('aws-track.png')
        s.textureData = pygame.image.tostring(s.img, "RGB", 1)
        s.aspect_ratio = fr_width/fr_height
        s.width = s.img.get_width()
        s.height = s.img.get_height()
        s.scl = scl
        r = s.width/s.height
        s.vert = s.scl*np.array([[-r, -1, 0],
                               [-r,  1, 0],
                               [ r,  1, 0],
                               [ r, -1, 0]], dtype=np.float32)

        s.im = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, s.im)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, s.width, s.height, 0, GL_RGB, GL_UNSIGNED_BYTE, s.textureData)
        glEnable(GL_TEXTURE_2D)

        glLoadIdentity()
        gluPerspective(45, s.aspect_ratio, 0.05, 100)
        s.translate(0,0,-scl/5)
        # s.draw()

    def wall(s):
        glBegin(GL_QUADS)
        glTexCoord2f(0,0)
        glVertex3f(*s.vert[0])
        glTexCoord2f(0,1)
        glVertex3f(*s.vert[1])
        glTexCoord2f(1,1)
        glVertex3f(*s.vert[2])
        glTexCoord2f(1,0)
        glVertex3f(*s.vert[3])
        glEnd()

    def translate_img(s, x, y, z=0):
        s.vert[:,:2] += np.array([x,y])*s.scl
        if z!=0: s.vert[:,2] += z*s.scl

    def translate(s, x, y, z=0):
        glTranslatef(x, y, z)

    def rotate_x(s, deg):
        glRotate(deg, 1, 0, 0)

    def rotate_y(s, deg):
        glRotate(deg, 0, 1, 0)

    def rotate_z(s, deg):
        glRotate(deg, 0, 0, 1)

    def draw(s):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        s.wall()
        pygame.display.flip()

if __name__ == "__main__":
    width, height = 600,600
    pygame.init()
    pygame.display.set_mode((width,height), DOUBLEBUF|OPENGL)
