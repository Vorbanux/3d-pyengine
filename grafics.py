import pygame
import engine as en
import numpy as n

base_color = None
screen = None

def create_window(width: int, height: int, bg_color: list):
    screen = pygame.display.set_mode((width, height))
    screen.display.set_caption("My 3D Engine")
    screen.fill(bg_color)

def edge(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float):
    return en.edge(n.array([[x1], [y1], [z1]], dtype=n.float32), n.array([[x2], [y2], [z2]], dtype=n.float32))

def start():
    while True:
        if not base_color:
            base_color = screen
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys
                sys.exit()
        pygame.display.flip()
    