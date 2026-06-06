import pygame
import engine as en
import numpy as n
import sys
import math

base_color = None
screen = None
cam = None
vertices = n.empty((0, 4), dtype=n.float32)
faces = []
faces_colors = n.empty((0, 4), dtype=n.int32)
faces_lines = n.empty((0, 5), dtype=n.int32)

def world_location(world_primary: list, tolist: bool, local_points: list) -> n.ndarray or list:
    world_points = n.array(local_points, dtype=n.float32)
    primary_matrix = n.ones((len(local_points), 3), dtype=n.float32) * world_primary
    world_points[:, :3] += primary_matrix[:, :3]
    if tolist: world_points = world_points.tolist()
    return world_points

def local_location(world_primary: list, tolist: bool, *world_points: list) -> n.ndarray or list:
    local_points = n.array(world_points, dtype=n.float32)
    primary_matrix = n.ones((len(world_points), 3), dtype=n.float32) * world_primary
    local_points[:, :3] -= primary_matrix[:, :3]
    if tolist: local_points = local_points.tolist()
    return local_points

def create_window(width: int, height: int, bg_color: list, cam_x: float, cam_y: float, cam_z: float, cam_angle: float = 70):
    global base_color, screen, cam
    pygame.font.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("My 3D Engine")
    base_color = bg_color
    screen.fill(bg_color)
    cam = en.camera(cam_x, cam_y, cam_z, cam_angle)
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

class face:
    def __init__(self, color: list, line_color: list, line_width: int, primary: list, world_poses: list = None, local_poses: list = None):
        global vertices, faces, faces_colors, faces_lines
        
        self.color = color
        self.line_color = line_color
        self.line_width = line_width
        self.primary = primary
        
        self.rotation = n.array([[0.0], [0.0], [0.0]], dtype=n.float32)
        
        if world_poses:
            self.world_poses = world_poses
            self.local_poses = local_location(primary, True, world_poses)[0]
        elif local_poses:

            self.world_poses = world_location(primary, True, local_poses)[0]
            self.local_poses = local_poses
        else:
            raise ValueError("error: world_poses or local_poses must contain value: list")
        
        poses = list(self.world_poses)[::-1]
        poses.append(primary)
        poses = poses[::-1]
        
        n_points = len(poses)
        pos = vertices.shape[0]
        
        new_verts = n.array([[float(p[0]), float(p[1]), float(p[2]), 1.0] for p in poses], dtype=n.float32)
        vertices = n.vstack((vertices, new_verts))
        
        poly_indices = list(range(pos, pos + n_points))
        
        faces.append(poly_indices)
        
        f_alpha = color[3] if len(color) > 3 else 255
        new_color = n.array([[color[0], color[1], color[2], f_alpha]], dtype=n.int32)
        faces_colors = n.vstack((faces_colors, new_color))
        
        l_alpha = line_color[3] if len(line_color) > 3 else 255
        
        faces_lines = n.vstack((faces_lines, n.array([[int(line_color[0]), int(line_color[1]), int(line_color[2]), int(l_alpha), int(line_width)]], dtype=n.int32)))

    def location(self): 
        self.world_poses = world_location(self.primary, True, self.local_poses)

    @property
    def get_location(self) -> list: return self.primary
    
    @get_location.setter
    def get_location(self, value: list): 
        self.primary = value
        self.location()

class edge:
    def __init__(self, color: list, width: list, primary: list, world_end: list = None, local_end: list = None):
        self.color = color
        self.width = width
        self.primary = primary
        
        if world_end:
            self.world_end = world_end
            self.local_end = local_location(primary, True, [world_end])
        elif local_end:
            self.world_end = world_location(primary, True, [local_end])
            self.local_end = local_end
        else:
            raise ValueError("error: world_poses or local_poses must contain value: list")
        
        face((0, 0, 0, 0), color, width, primary, self.world_end)
    
    def location(self): 
        self.world_end = world_location(self.primary, True, self.local_end)

    @property
    def get_location(self) -> list: return self.primary
    
    @get_location.setter
    def get_location(self, value: list): 
        self.primary = value
        self.location()


class cube:
    def __init__(self, color: list, line_color: list, line_width: int, primary: list, world_poses: list = None, local_poses: list = None):
        self.color = color
        self.line_color = line_color
        self.line_width = line_width
        self.primary = primary
        
        if world_poses:
            self.world_poses = world_poses
            self.local_poses = local_location(primary, True, world_poses)
        elif local_poses:

            self.world_poses = world_location(primary, True, local_poses)
            self.local_poses = local_poses
        else:
            raise ValueError("error: world_poses or local_poses must contain value: list")
        
        poses = list(self.world_poses)[::-1]
        poses.append(primary)
        poses = poses[::-1]
        
        cube_topology = [
            [0, 1, 2, 3],  # Down
            [2, 3, 7, 6],  # Back
            [3, 0, 4, 7],  # Left
            [0, 1, 5, 4],  # Forward
            [1, 2, 6, 5],  # Right
            [4, 5, 6, 7]   # Up
        ]

        p = poses
        for face_indices in cube_topology:

            face_verts = [p[idx] for idx in face_indices]
            face(
                color, 
                line_color, 
                line_width, 
                primary=face_verts[0], 
                world_poses=face_verts[1:]
            )
    
    def location(self): 
        self.world_poses = world_location(self.primary, True, self.local_poses)

    @property
    def get_location(self) -> list: return self.primary
    
    @get_location.setter
    def get_location(self, value: list): 
        self.primary = value
        self.location()


def pyramid(line_color: list, line_width: float, color: list, pos_primary: list, poses: list):
    face(color, line_color, line_width, poses[0], poses[1:])
    for p in range(len(poses)):
        next = (p+1) % len(poses)
        face(color, line_color, line_width, poses[p], (poses[next], pos_primary))


def update():
    global vertices, edges, colors, faces, faces_colors, faces_lines
    result = cam.projection(faces, vertices, [screen.get_width(), screen.get_height()])
    if result is not None:
        final, valid, avg = result
        if isinstance(valid, tuple):
            valid = valid[0]
        sorted_indices = n.argsort(-avg)
        for i in sorted_indices:
            original_face_idx = valid[i]
            if len(final[i]) < 3:
                continue
            color_rgb = tuple(faces_colors[original_face_idx, :3])
            pygame.draw.polygon(screen, color_rgb, final[i])
            l_color_rgb = (
                int(faces_lines[original_face_idx, 0]), 
                int(faces_lines[original_face_idx, 1]), 
                int(faces_lines[original_face_idx, 2])
            )
            l_width = int(faces_lines[original_face_idx, 4])
            pygame.draw.polygon(screen, l_color_rgb, final[i], l_width)
    
        text_surface = pygame.font.SysFont(None, 24).render(f"x: {math.floor(cam.pos[0, 0]*100)/100}; y: {math.floor(cam.pos[1, 0]*100)/100}; z: {math.floor(cam.pos[2, 0]*100)/100}", True, (255, 0, 0, 1))

        screen.blit(text_surface, (100, 100))
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
        
        move_speed = 0.003
        
        d_yaw = 0.0
        d_pitch = 0.0
        
        mouse_sensitivity = 0.0015
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
            move_vec = (move_vec / n.linalg.norm(move_vec)) * move_speed

            cam.x += move_vec[0]
            cam.z += move_vec[2]
            
        if dx != 0 or dy != 0:
            d_yaw = -dx * mouse_sensitivity
            d_pitch = -dy * mouse_sensitivity  
            
            cam.rotate(d_yaw, d_pitch)

        screen.fill(base_color)
        update()

        pygame.display.flip()