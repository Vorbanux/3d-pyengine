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

class camera(point):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0, cm_angle: float = 70):
        super().__init__(x, y, z, (0, 0, 255, 0))
        self.lookvector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.upvector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.rightvector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.position = np.array([
           [1.0, 0.0, 0.0, -self.pos[0, 0]],
           [0.0, 1.0, 0.0, -self.pos[1, 0]],
           [0.0, 0.0, 1.0, -self.pos[2, 0]],
           [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        self.rotation = np.array([
            [self.rightvector[0], self.rightvector[1], self.rightvector[2], 0.0],
            [self.upvector[0], self.upvector[1], self.upvector[2], 0],
            [self.lookvector[0], self.lookvector[1], self.lookvector[2], 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        self.angle = cm_angle
        
        _yaw = np.arctan2(self.lookvector[0], self.lookvector[2])
        _pitch = np.arctan2(self.lookvector[1], np.sqrt(self.lookvector[0]**2 + self.lookvector[2]**2))
        _roll = np.arctan2(np.dot(self.upvector, self.rightvector), np.dot(self.upvector, np.array([0, 1, 0])))
        
        self.transform = self.rotation @ self.position
    
    def projection(self, data_point: np.ndarray, vertices: np.ndarray, screen_size: list) -> list: #[[point1, point2], [point2, point3], ..]
        
        self.position = np.array([
           [1.0, 0.0, 0.0, -self.pos[0, 0]],
           [0.0, 1.0, 0.0, -self.pos[1, 0]],
           [0.0, 0.0, 1.0, -self.pos[2, 0]],
           [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        self.rotation = np.array([
            [self.rightvector[0], self.upvector[0], self.lookvector[0], 0.0],
            [self.rightvector[1], self.upvector[1], self.lookvector[1], 0.0],
            [self.rightvector[2], self.upvector[2], self.lookvector[2], 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        self.transform = self.rotation @ self.position
        
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
        
        idx_start = data_point[:, 0]
        idx_end   = data_point[:, 1]
        
        p1 = clip_points[idx_start]
        p2 = clip_points[idx_end]

        w1 = p1[:, 3]
        w2 = p2[:, 3]

        both_behind = (w1 < NEAR_PLANE) & (w2 < NEAR_PLANE)
        both_too_far = (w1 > FAR_PLANE) & (w2 > FAR_PLANE)

        culled_mask = both_behind | both_too_far

        valid_indices = np.where(~culled_mask)[0]
        p1_valid = p1[valid_indices].copy()
        p2_valid = p2[valid_indices].copy()
        w1_valid = w1[valid_indices]
        w2_valid = w2[valid_indices]

        t1 = (NEAR_PLANE - w1_valid) / (w2_valid - w1_valid + 1e-6)
        mask1 = (w1_valid < NEAR_PLANE) & (w2_valid >= NEAR_PLANE)
        p1_valid[mask1] = p1_valid[mask1] + t1[mask1, np.newaxis] * (p2_valid[mask1] - p1_valid[mask1])
        
        t2 = (NEAR_PLANE - w2_valid) / (w1_valid - w2_valid + 1e-6)
        mask2 = (w2_valid < NEAR_PLANE) & (w1_valid >= NEAR_PLANE)
        p2_valid[mask2] = p2_valid[mask2] + t2[mask2, np.newaxis] * (p1_valid[mask2] - p2_valid[mask2])
        
        t3 = (FAR_PLANE - w1_valid) / (w2_valid - w1_valid + 1e-6)
        mask3 = (w1_valid > FAR_PLANE) & (w2_valid <= FAR_PLANE)
        p1_valid[mask3] = p1_valid[mask3] + t3[mask3, np.newaxis] * (p2_valid[mask3] - p1_valid[mask3])
        
        t4 = (FAR_PLANE - w2_valid) / (w1_valid - w2_valid + 1e-6)
        mask4 = (w2_valid > FAR_PLANE) & (w1_valid <= FAR_PLANE)
        p2_valid[mask4] = p2_valid[mask4] + t4[mask4, np.newaxis] * (p1_valid[mask4] - p2_valid[mask4])

        p1_ndc = p1_valid[:, :2] / p1_valid[:, 3:4]
        p2_ndc = p2_valid[:, :2] / p2_valid[:, 3:4]
        
        width, height = screen_size[0], screen_size[1]

        scr_x1 = ((p1_ndc[:, 0] + 1.0) * width / 2.0).astype(np.int32)
        scr_y1 = ((-p1_ndc[:, 1] + 1.0) * height / 2.0).astype(np.int32)

        scr_x2 = ((p2_ndc[:, 0] + 1.0) * width / 2.0).astype(np.int32)
        scr_y2 = ((-p2_ndc[:, 1] + 1.0) * height / 2.0).astype(np.int32)
        lines_start = np.column_stack((scr_x1, scr_y1))
        lines_end = np.column_stack((scr_x2, scr_y2))
        
        return lines_start, lines_end, valid_indices

        
    
    @property
    def x(self) -> float:
        return self.pos[0, 0]
        
    @property
    def y(self) -> float:
        return self.pos[1, 0]
        
    @property
    def z(self) -> float:
        return self.pos[2, 0]
        
    @property
    def yaw(self) -> float:
        return self._yaw
        
    @property
    def pitch(self) -> float:
        return self._pitch
    
    @property
    def row(self) -> float:
        return self._row
        
    @property
    def yaw_deg(self) -> float:
        return float(np.degrees(self._yaw))
        
    @property
    def pitch_deg(self) -> float:
        return float(np.degrees(self._pitch))
    
    @property
    def row_deg(self) -> float:
        return float(np.degrees(self._row))
  
    @x.setter
    def x(self, value: float): self.pos[0, 0] = value
        
    @y.setter
    def y(self, value: float): self.pos[1, 0] = value
        
    @z.setter
    def z(self, value: float): self.pos[2, 0] = value
        
    @yaw.setter
    def yaw(self, value: float): self._yaw = value
        
    @pitch.setter
    def pitch(self, value: float): self._pitch = value
    
    @row.setter
    def row(self, value: float): self._row = value

    @yaw_deg.setter
    def yaw(self, value: float): self._yaw = np.radian(value)
        
    @pitch_deg.setter
    def pitch(self, value: float): self._pitch = np.radian(value)
    
    @row_deg.setter
    def row(self, value: float): self._row = np.radian(value)