import pygame
import engine as en
import numpy as n

base_color = None
screen = None
cam = None
vertices = n.empty((0, 4), dtype=n.float32)
edges = n.empty((0, 2), dtype=n.int32)
colors = n.empty((0, 4), dtype=n.int32)

def create_window(width: int, height: int, bg_color: list, cam_x: float, cam_y: float, cam_z: float, cam_angle: float = 70):
    global base_color, screen, cam
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("My 3D Engine")
    base_color = bg_color
    screen.fill(bg_color)
    cam = en.camera(cam_x, cam_y, cam_z, cam_angle)

def edge(pos_primary: list, pos_end: list, color: list):
    global vertices, edges, colors
    start = n.array([[float(pos_primary[0]), float(pos_primary[1]), float(pos_primary[2]), 1.0]], dtype=n.float32)
    end = n.array([[float(pos_end[0]), float(pos_end[1]), float(pos_end[2]), 1.0]], dtype=n.float32)
    pos = vertices.shape[0]
    vertices = n.vstack((vertices, start))
    vertices = n.vstack((vertices, end))
    edges = n.vstack((edges, n.array([[pos, pos + 1]], dtype=n.int32)))
    colors = n.vstack((colors, n.array([[color[0], color[1], color[2], color[3]]], dtype=n.int32)))

def face(pos_primary: list, pos_middle: list, pos_end: list, color: list):
    global vertices, edges, colors
    edge(pos_primary, pos_middle, color)
    edge(pos_middle, pos_end, color)
    edge(pos_end, pos_primary, color)

def update():
    global vertices, edges, colors
    result = cam.projection(edges, vertices, [screen.get_width(), screen.get_height()])
    if result is not None:
        lines_start, lines_end, valid_indices = result
        if isinstance(valid_indices, tuple):
            valid_indices = valid_indices[0]
        for i in range(len(lines_start)):
            p1 = tuple(lines_start[i])
            p2 = tuple(lines_end[i])
            original_edge_idx = valid_indices[i]
            color_rgb = tuple(colors[original_edge_idx, :3])
            pygame.draw.line(screen, color_rgb, p1, p2, 2)
        return True
    

def start():
    global base_color, screen
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys
                sys.exit()
        
        keys = pygame.key.get_pressed()
    
        speed = 0.001
    
        if keys[pygame.K_w]:
            cam.z += speed
        if keys[pygame.K_s]:
            cam.z -= speed
        if keys[pygame.K_a]:
            cam.x += speed
        if keys[pygame.K_d]:
            cam.x -= speed

        screen.fill(base_color)
        update()

        pygame.display.flip()