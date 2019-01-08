import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import matplotlib.pyplot as plt

class Display:
    def __init__(s, fr_height, fr_width, filename=None, img=None):
        if filename is not None:
            s.img = pygame.image.load(filename)
        else:
            s.img = pygame.surfarray.make_surface(np.swapaxes(img,0,1))
        s.textureData = pygame.image.tostring(s.img, "RGB", 1)
        s.aspect_ratio = fr_width/fr_height
        s.width = s.img.get_width()
        s.height = s.img.get_height()
        s.z_angle = 0

        # 3d coordinates of image
        s.vert = np.array([[0,       0,        0],
                           [0,       s.height, 0],
                           [s.width, s.height, 0],
                           [s.width, 0,        0]], dtype=np.float32)

        #opengl boilerplate code
        s.im = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, s.im)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, s.width, s.height, 0, GL_RGB, GL_UNSIGNED_BYTE, s.textureData)
        glEnable(GL_TEXTURE_2D)

        glLoadIdentity()
        gluPerspective(45, s.aspect_ratio, 0.05, 10000)
        
        #initial adjustments
        s.translate(0,0,-s.height/10) #lift the camera slightly

        
#     def new_track(s, numpy_img):
#         surface_img = pygame.surfarray.make_surface(numpy_img)
#         textureData = pygame.image.tostring(surface_img, "RGB", 1)
#         glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, s.width, s.height, 0, GL_RGB, GL_UNSIGNED_BYTE, textureData)
        
#     def reset_track(s):
#         glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, s.width, s.height, 0, GL_RGB, GL_UNSIGNED_BYTE, s.textureData)
        
    def wall(s):
        #project the image onto 3d space
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
        #translate the image, not the camera
        s.vert[:,:2] += np.array([-x,-y])
        if z!=0: s.vert[:,2] += z

    def move_img_to(s, x, y, z=0):
        #specify new image location
        s.vert[:,:2] = np.array([[-x,        -y,],
                                 [-x,        s.height-y],
                                 [s.width-x, s.height-y],
                                 [s.width-x, -y]], dtype=np.float32)
        if z!=0: s.vert[:,2] = z

    def translate(s, x, y, z=0):
        #translate the camera
        glTranslatef(x, y, z)

    def rotate_x(s, deg):
        #rotate the camera
        glRotate(deg, 1, 0, 0)

#     def rotate_y(s, deg):
#         #rotate the camera
#         glRotate(deg, 0, 1, 0)

#     def rotate_z(s, deg):
#         #rotate the camera
#         s.z_angle += deg
#         glRotate(deg, 0, 0, 1)

    def rotate_z_abs(s, deg):
        #rotate the camera
        deg += 90 # plus 90 because ...
        glRotate(deg-s.z_angle, 0, 0, 1)
        s.z_angle = deg

    def draw(s):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        s.wall()
        pygame.display.flip()

    def read_screen(s):
        x, y, width, height = glGetIntegerv(GL_VIEWPORT)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = np.empty(width*height*3, dtype=np.uint8)
        glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE, data)
        return data.reshape(height, width, 3)[::-1].copy()

if __name__ == "__main__":
    width, height = 1000,600
    pygame.init()
    pygame.display.set_mode((width,height), DOUBLEBUF|OPENGL)
