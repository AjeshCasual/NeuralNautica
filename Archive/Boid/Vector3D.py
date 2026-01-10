class Vector3D:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def magnitude(self):
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return (
            Vector3D(self.x / mag, self.y / mag, self.z / mag)
            if mag > 0
            else Vector3D(0, 0, 0)
        )
