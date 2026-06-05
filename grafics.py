import pygame
import engine as en
import numpy as n
import sys

base_color = None
screen = None
cam = None
vertices = n.empty((0, 4), dtype=n.float32)
edges = n.empty((0, 2), dtype=n.int32)
faces = n.empty((0, 4), dtype=n.int32)
faces_colors = n.empty((0, 4), dtype=n.int32)
colors = n.empty((0, 4), dtype=n.int32)
faces_lines = n.empty((0, 5), dtype=n.int32)

def create_window(width: int, height: int, bg_color: list, cam_x: float, cam_y: float, cam_z: float, cam_angle: float = 70):
    global base_color, screen, cam
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("My 3D Engine")
    base_color = bg_color
    screen.fill(bg_color)
    cam = en.camera(cam_x, cam_y, cam_z, cam_angle)
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

def edge(pos_primary: list, pos_end: list, color: list):
    global vertices, edges, colors
    start = n.array([[float(pos_primary[0]), float(pos_primary[1]), float(pos_primary[2]), 1.0]], dtype=n.float32)
    end = n.array([[float(pos_end[0]), float(pos_end[1]), float(pos_end[2]), 1.0]], dtype=n.float32)
    pos = vertices.shape[0]
    vertices = n.vstack((vertices, start))
    vertices = n.vstack((vertices, end))
    edges = n.vstack((edges, n.array([[pos, pos + 1]], dtype=n.int32)))
    colors = n.vstack((colors, n.array([[color[0], color[1], color[2], color[3]]], dtype=n.int32)))

def face(color: list, line_color: list, line_width: int, *poses: list):
    global vertices, faces, faces_colors, faces_lines
    n_points = len(poses)
    pos = vertices.shape[0]
    
    new_verts = n.array([[float(p[0]), float(p[1]), float(p[2]), 1.0] for p in poses], dtype=n.float32)
    
    poly_indices = n.array([list(range(pos, pos + n_points))], dtype=n.int32)
    
    vertices = n.vstack((vertices, new_verts))
    faces = n.vstack((faces, poly_indices))
    new_color = n.array([[color[0], color[1], color[2], color[3]]], dtype=n.int32)
    faces_colors = n.vstack((faces_colors, new_color))
    faces_lines = n.vstack((faces_lines, n.array([[line_color[0], line_color[1], line_color[2], line_color[3], line_width]], dtype=n.int32)))

def cube(line_color: list, line_width: float, pos_primary: list, pos_B: list, pos_C: list, pos_D: list, pos_A1: list, pos_B1: list, pos_C1: list, pos_D1: list, color: list):
    face(color, line_color, line_width, pos_primary, pos_B, pos_C, pos_D)
    face(color, line_color, line_width, pos_C, pos_D, pos_D1, pos_C1)
    face(color, line_color, line_width, pos_D, pos_primary, pos_A1, pos_D1)
    face(color, line_color, line_width, pos_primary, pos_B, pos_B1, pos_A1)
    face(color, line_color, line_width, pos_B, pos_C, pos_C1, pos_B1)
    face(color, line_color, line_width, pos_A1, pos_B1, pos_C1, pos_D1)

def update():
    global vertices, edges, colors, faces, faces_colors, faces_lines
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
    result = cam.projection_faces(faces, vertices, [screen.get_width(), screen.get_height()])
    if result is not None:
        final, valid, avg = result
        if isinstance(valid, tuple):
            valid = valid[0]
        sorted_indices = n.argsort(-avg)
        for i in sorted_indices:
            original_face_idx = valid[i]
            color_rgb = tuple(faces_colors[original_face_idx, :3])
            pygame.draw.polygon(screen, color_rgb, final[i])
            pygame.draw.polygon(screen, (int(faces_lines[original_face_idx, :3][0]), int(faces_lines[original_face_idx, :3][1]), int(faces_lines[original_face_idx, :3][2])), final[i], faces_lines[original_face_idx][4])
        return True

def start():
    global base_color, screen, sys
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys
                sys.exit()
        
        keys = pygame.key.get_pressed()
        
        move_speed = 0.001
        
        d_yaw = 0.0
        d_pitch = 0.0
        
        mouse_sensitivity = 0.0015  # Чувствительность мыши
        dx, dy = pygame.mouse.get_rel()

        move_vec = n.zeros(3, dtype=n.float32)
        
        if keys[pygame.K_w]:
            move_vec += cam.lookvector
        if keys[pygame.K_s]:
            move_vec -= cam.lookvector
        if keys[pygame.K_a]:
            move_vec += cam.rightvector
        if keys[pygame.K_d]:
            move_vec -= cam.rightvector
        if keys[pygame.K_ESCAPE]:
            sys.exit()
        
        if n.any(move_vec):
            # Нормализуем вектор движения, чтобы скорость по диагонали не была быстрее
            move_vec = (move_vec / n.linalg.norm(move_vec)) * move_speed
            
            # Прибавляем смещение к координатам камеры
            cam.x += move_vec[0]
            cam.z += move_vec[2]
            
        if dx != 0 or dy != 0:
            d_yaw = dx * mouse_sensitivity
            d_pitch = -dy * mouse_sensitivity  
            
            cam.rotate(d_yaw, d_pitch)

        screen.fill(base_color)
        update()

        pygame.display.flip()