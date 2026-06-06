FAR_PLANE = 100
NEAR_PLANE = 0.1

import numpy as np


class point:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, color: list = [255, 255, 255, 1]):
        self.pos = np.array([
            [x], [y], [z], [1.0]
        ], dtype=np.float32)
        self.rgba = color

    @property
    def x(self) -> float:
        return self.pos[0, 0]
    
    @property
    def y(self) -> float:
        return self.pos[1, 0]

    @property
    def z(self) -> float:
        return self.pos[2, 0]
    
    @x.setter
    def x(self, value: float): self.pos[0, 0] = value
        
    @y.setter
    def y(self, value: float): self.pos[1, 0] = value
        
    @z.setter
    def z(self, value: float): self.pos[2, 0] = value
    
    def local_transform(self, parent: point):
        self.parent = parent
        dpos = self.pos - parent.pos
        self.local_pos = np.array([[dpos[0][0]], [dpos[1][0]], [dpos[2][0]], [1.0]], dtype=np.float32)
        return self.local_pos

import numpy as np

FAR_PLANE = 100
NEAR_PLANE = 0.1

def clip_polygon_near(poly_vertices: np.ndarray, near: float):
    if len(poly_vertices) < 3:
        return np.empty((0, 4), dtype=np.float32)
    output_vertices = []
    s = poly_vertices[-1]
    for e in poly_vertices:
        s_inside = s[3] >= near
        e_inside = e[3] >= near
        if e_inside:
            if not s_inside:
                t = (near - s[3]) / (e[3] - s[3] + 1e-6)
                intersect_point = s + t * (e - s)
                output_vertices.append(intersect_point)
            output_vertices.append(e)
        else:
            if s_inside:
                t = (near - s[3]) / (e[3] - s[3] + 1e-6)
                intersect_point = s + t * (e - s)
                output_vertices.append(intersect_point)
        s = e
    return np.array(output_vertices, dtype=np.float32)

class camera(point):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0, cm_angle: float = 70):
        super().__init__(x, y, z, (0, 0, 255, 0))
        self.angle = cm_angle
        self._yaw = 0.0
        self._pitch = 0.0
        
        self.update_vectors()

    def update_vectors(self):
        self._pitch = np.clip(self._pitch, np.radians(-89.0), np.radians(89.0))
        look_x = np.sin(self._yaw) * np.cos(self._pitch)
        look_y = np.sin(self._pitch)
        look_z = np.cos(self._yaw) * np.cos(self._pitch)
        
        self.lookvector = np.array([look_x, look_y, look_z], dtype=np.float32)
        self.lookvector /= np.linalg.norm(self.lookvector)
        
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.rightvector = np.cross(world_up, self.lookvector)
        self.rightvector /= np.linalg.norm(self.rightvector)
        self.upvector = np.cross(self.lookvector, self.rightvector)
        self.upvector /= np.linalg.norm(self.upvector)

    def rotate(self, delta_yaw: float, delta_pitch: float):
        self._yaw += delta_yaw
        self._pitch += delta_pitch
        self.update_vectors()

    def _update_matrices(self):
        self.position = np.array([
           [1.0, 0.0, 0.0, -self.pos[0, 0]],
           [0.0, 1.0, 0.0, -self.pos[1, 0]],
           [0.0, 0.0, 1.0, -self.pos[2, 0]],
           [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.rotation = np.array([
            [self.rightvector[0], self.rightvector[1], self.rightvector[2], 0.0],
            [self.upvector[0],    self.upvector[1],    self.upvector[2],    0.0],
            [self.lookvector[0],  self.lookvector[1],  self.lookvector[2],  0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        self.transform = self.rotation @ self.position

    def projection(self, data_point: np.ndarray, vertices: np.ndarray, screen_size: list):
        self._update_matrices()
        
        f = 1 / np.tan(np.radians(self.angle)/2)
        aspect = screen_size[0] / screen_size[1]
        project = np.array([
                [-f/aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (FAR_PLANE+NEAR_PLANE)/(NEAR_PLANE-FAR_PLANE), (2*FAR_PLANE*NEAR_PLANE)/(NEAR_PLANE-FAR_PLANE)],
                [0.0, 0.0, -1.0, 0.0]
            ], dtype=np.float32)
        mvp = project @ self.transform
        
        clip_points = (mvp @ vertices.T).T
        clip_points[:, 3] = -clip_points[:, 3]
        
        final_polygons = []
        valid_indices = []
        avg_depths = []
        
        width, height = screen_size[0], screen_size[1]
        
        for face_idx, face_vertices in enumerate(data_point):
            poly_verts = clip_points[face_vertices]
            clipped_verts = clip_polygon_near(poly_verts, NEAR_PLANE)
            if len(clipped_verts) < 3:
                continue
                
            ndc = clipped_verts[:, :2] / clipped_verts[:, 3:4]
            scr_x = ((ndc[:, 0] + 1.0) * width / 2.0).astype(np.int32)
            scr_y = ((-ndc[:, 1] + 1.0) * height / 2.0).astype(np.int32)
            
            screen_poly = np.column_stack((scr_x, scr_y))
            final_polygons.append(screen_poly)
            valid_indices.append(face_idx)
            avg_depths.append(np.mean(clipped_verts[:, 3]))
            
        return final_polygons, valid_indices, np.array(avg_depths, dtype=np.float32)
    
    @property
    def x(self) -> float:
        return self.pos[0, 0]
    
    @property
    def y(self) -> float:
        return self.pos[1, 0]

    @property
    def z(self) -> float:
        return self.pos[2, 0]
    
    @x.setter
    def x(self, value: float): self.pos[0, 0] = value
        
    @y.setter
    def y(self, value: float): self.pos[1, 0] = value
        
    @z.setter
    def z(self, value: float): self.pos[2, 0] = value
