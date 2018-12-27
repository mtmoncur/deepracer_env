import pygame, OpenGL
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

class Display:
    def __init__(s, fr_height, fr_width, scl, filename):
        s.img = pygame.image.load('aws-track.png')
        s.textureData = pygame.image.tostring(s.img, "RGB", 1)
        s.aspect_ratio = fr_width/fr_height
        s.width = s.img.get_width()
        s.height = s.img.get_height()
        s.scl = scl
        r = s.width/s.height

        # 3d coordinates of image
#         s.vert = s.scl*np.array([[-r, -1, 0],
#                                 [-r,  1, 0],
#                                 [ r,  1, 0],
#                                 [ r, -1, 0]], dtype=np.float32)
        s.vert = s.scl*np.array([[-s.width/2, -s.height/2, 0],
                                 [-s.width/2,  s.height/2, 0],
                                 [ s.width/2,  s.height/2, 0],
                                 [ s.width/2, -s.height/2, 0]], dtype=np.float32)

        #opengl boilerplate code
        s.im = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, s.im)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, s.width, s.height, 0, GL_RGB, GL_UNSIGNED_BYTE, s.textureData)
        glEnable(GL_TEXTURE_2D)

        glLoadIdentity()
        gluPerspective(45, s.aspect_ratio, 0.05, 10000)
        s.translate(0,0,-scl*s.height/10) #lift the camera slightly
#         gluPerspective(45, s.aspect_ratio, 0.05, 100)
#         s.translate(0,0,-scl/5) #lift the camera slightly

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
        s.vert[:,:2] += np.array([x,y])*s.scl
        if z!=0: s.vert[:,2] += z*s.scl

    def move_img_to(s, x, y, z=0):
        #specify new image location
        r = s.width/s.height
        s.vert[:,:2] = s.scl*np.array([[-s.width+x, -s.height+y, 0],
                                       [-s.width+x,  s.height+y, 0],
                                       [ s.width+x,  s.height+y, 0],
                                       [ s.width+x, -s.height+y, 0]], dtype=np.float32)
        if z!=0: s.vert[:,2] = z*s.scl

    def translate(s, x, y, z=0):
        #translate the camera
        glTranslatef(x, y, z)

    def rotate_x(s, deg):
        #rotate the camera
        glRotate(deg, 1, 0, 0)

    def rotate_y(s, deg):
        #rotate the camera
        glRotate(deg, 0, 1, 0)

    def rotate_z(s, deg):
        #rotate the camera
        glRotate(deg, 0, 0, 1)

    def draw(s):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        s.wall()
        pygame.display.flip()

    def read_screen(s):
        import cupy as cp
        x, y, width, height = glGetIntegerv(GL_VIEWPORT)
        # print("Screenshot viewport:", x, y, width, height)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)

        data = np.empty(width*height*3, dtype=np.uint8)
        # print(type(width*height*3))
        # data = cp.empty(int(width*height*3), dtype=cp.uint8)
        #data = torch.empty(width*height*3, dtype=torch.uint8, device='cuda')
        glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE, data)
        # image = np.frombuffer(data, dtype=np.uint8).reshape(height,width,3)
        # plt.imshow(image)
        # plt.show()
        return data.reshape(height, width, 3)[::-1]

if __name__ == "__main__":
    width, height = 600,600
    pygame.init()
    pygame.display.set_mode((width,height), DOUBLEBUF|OPENGL)
