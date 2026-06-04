FAR_PLANE = 100
NEAR_PLANE = 0.1

import numpy as np
class point:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, red: int = 255, green: int = 255, blue: int = 255, alpha: float = 1.0):
        self.pos = np.array([
            [x], [y], [z], [1.0]
        ], dtype=np.float32)
        self.rgba = [red, green, blue, alpha]
        print(self.pos)
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

class edge:
    def __init__(self, point1pos: np.ndarray, point2pos: np.ndarray):
        self.primary = point(point1pos[0, 0], point1pos[1, 0], point1pos[2, 0])
        self.end = point(point2pos[0, 0], point2pos[1, 0], point2pos[2, 0])
        self.position = np.array([
            [1.0, 0.0, 0.0, point1pos[0][0]],
            [0.0, 1.0, 0.0, point1pos[1][0]],
            [0.0, 0.0, 1.0, point1pos[2][0]],
            [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32)
        
        dx = point2pos[0][0]-point1pos[0][0]
        dy = point2pos[1][0]-point1pos[1][0]
        dz = point2pos[2][0]-point1pos[2][0]
        
        length = (dx**2 + dy**2 + dz**2)**0.5

        if length == 0:
            length = 1e-6

        self.size = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, length, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        forward = np.array([dx, dy, dz]) / length
        up_tmp = np.array([0.0, 1.0, 0.0]) if abs(forward[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
        right = np.cross(up_tmp, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        
        self._yaw = np.arctan2(dx, dz)
        
        horizontal_dist = (dx**2 + dz**2)**0.5        
        if horizontal_dist == 0:
            self._pitch = np.pi / 2 if dy > 0 else -np.pi / 2
        else:
            self._pitch = np.arctan2(dy, horizontal_dist)
        
        self.rotation = np.array([
            [right[0], up[0], forward[0], 0.0],
            [right[1], up[1], forward[1], 0.0],
            [right[2], up[2], forward[2], 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        self.trasnsform = self.position @ self.rotation @ self.size
        
    @property
    def x(self) -> float:
        return self.primary.pos[0, 0]
        
    @property
    def y(self) -> float:
        return self.primary.pos[1, 0]
        
    @property
    def z(self) -> float:
        return self.primary.pos[2, 0]
        
    @property
    def yaw(self) -> float:
        return self._yaw
        
    @property
    def pitch(self) -> float:
        return self._pitch
        
    @property
    def yaw_deg(self) -> float:
        return float(np.degrees(self._yaw))
        
    @property
    def pitch_deg(self) -> float:
        return float(np.degrees(self._pitch))
        
    @x.setter
    def x(self, value: float): self.primary.pos[0, 0] = value
        
    @y.setter
    def y(self, value: float): self.primary.pos[1, 0] = value
        
    @x.setter
    def z(self, value: float): self.primary.pos[2, 0] = value
        
    @yaw.setter
    def _yaw(self, value: float): self._yaw = value
        
    @pitch.setter
    def _pitch(self, value: float): self._pitch = value
        
    @yaw_deg.setter
    def _yaw(self, value: float): self._yaw = np.radians(value)
        
    @pitch_deg.setter
    def _pitch(self, value: float): self._pitch = np.radians(value)



class camera(point):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0, cm_angle: float = 70):
        super().__init__(x, y, z, 0, 0, 255, 0)
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
        
        self.transform = self.rotation @ self.position
    
    def projection(self, obj, screen_size: list) -> list:
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
        
        self.transform = self.rotation @ self.position

        f = 1 / np.tan(np.radians(self.angle)/2)
        aspect = screen_size[0] / screen_size[1]
        project = np.array([
                [f/aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (FAR_PLANE+NEAR_PLANE)/(NEAR_PLANE-FAR_PLANE), (2*FAR_PLANE*NEAR_PLANE)/(NEAR_PLANE-FAR_PLANE)],
                [0.0, 0.0, -1.0, 0.0]
            ], dtype=np.float32)
        mvp = project @ self.transform @ obj.transform
        
        local_primary = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float32)
        local_end = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
        
        result_primary = mvp @ local_primary
        result_end = mvp @ local_end

        if result_primary[3, 0] == 0: result_primary[3, 0] = 1e-6
        if result_end[3, 0] == 0: result_end[3, 0] = 1e-6
        
        screenx_primary = result_primary[0, 0] / result_primary[3, 0]
        screeny_primary = result_primary[1, 0] / result_primary[3, 0]
        
        screenx_end = result_end[0, 0] / result_end[3, 0]
        screeny_end = result_end[1, 0] / result_end[3, 0]
        
        px_x1 = int((screenx_primary + 1.0) * screen_size[0] / 2.0)
        px_y1 = int((-screeny_primary + 1.0) * screen_size[1] / 2.0)
        px_x2 = int((screenx_end + 1.0) * screen_size[0] / 2.0)
        px_y2 = int((-screeny_end + 1.0) * screen_size[1] / 2.0)
        
        return [(px_x1, px_y1), (px_x2, px_y2)]
        