import pygame
import engine as en
import numpy as n

base_color = None
screen = None
cam = None
all_obj = []

def create_window(width: int, height: int, bg_color: list, cam_x: float, cam_y: float, cam_z: float, cam_angle: float = 70):
    global base_color, screen, cam
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("My 3D Engine")
    base_color = bg_color
    screen.fill(bg_color)
    cam = en.camera(cam_x, cam_y, cam_z, cam_angle)

def edge(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, red: int, green: int, blue: int, alpha: float) -> en.edge:
    obj = en.edge(n.array([[x1], [y1], [z1]], dtype=n.float32), n.array([[x2], [y2], [z2]], dtype=n.float32), red, green, blue, alpha)
    all_obj.append(obj)
    return obj

def update():
    for obj in all_obj:
        attributes = cam.projection(obj, [screen.get_width(), screen.get_height()])
        pygame.draw.line(screen, attributes[2], attributes[0], attributes[1], 1)
        
    

def start():
    global base_color, screen
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys
                sys.exit()
              
        screen.fill(base_color)
        update()

        pygame.display.flip()