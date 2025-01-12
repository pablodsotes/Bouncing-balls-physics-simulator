""""crear diccionario para poder debuggear el codigo gpu"""

import numpy as np
import pygame

V = pygame.Vector2

a = np.random

balls_dict = {}

class Ball():
    """define a ball with all its attributes"""

    def __init__(self, **kwargs):
        self.grabed = False
        self.number = kwargs.get("count")
        self.selected = False

        self.radius = kwargs.get("radius")
        self.p = V(kwargs.get("p"))
        self.v = V(kwargs.get("v"))
        self.att = V(kwargs.get("att"))
        self.mass = 3.14*self.radius**2
        self.rect = pygame.Rect((0, 0, 2 * self.radius, 2 * self.radius))
        self.rect.center = self.p
        1+1


cantidad_bolas=100_000
# cantidad_bolas= 640
n= np.sqrt(cantidad_bolas)

for i in range(cantidad_bolas):
    ball = Ball(count=i, radius=np.random.uniform(100/n,200/n) ,
                p = V(np.random.uniform(0,800),np.random.uniform(0,800)),
                v = V(np.random.uniform(-5,5),np.random.uniform(-5,5)),
                att = V(np.random.uniform(-5,5),np.random.uniform(-5,5)))

    balls_dict[i] = ball

print(balls_dict[5].v)
