class Vector2D:
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y

    def __str__(self):
        return f"X:{self.x} and Y:{self.y}"

    def __mul__(self, scalar: float):
        """Multiply vector by a scalar (v * k)."""
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float):
        """Multiply scalar by vector (k * v)."""
        return self.__mul__(scalar)

    def __add__(self, other):
        """Add two vectors (v1 + v2)."""
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """Subtract two vectors (v1 - v2)."""
        return Vector2D(self.x - other.x, self.y - other.y)

    def magnitude(self):
        """Length of the vector."""
        return (self.x**2 + self.y**2) ** 0.5

    def normalize(self):
        """Return a unit vector in the same direction."""
        mag = self.magnitude()
        if mag > 0:
            return Vector2D(self.x / mag, self.y / mag)
        return Vector2D(0, 0)
