Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]

# Basic 2D vector helpers
def add_vec2(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])

def sub_vec2(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])

def mul_scalar_vec2(a: Vec2, s: float) -> Vec2:
    return (a[0] * s, a[1] * s)

def dot_vec2(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]
