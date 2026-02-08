"""
Physical constants for projectile motion simulations.
"""
import math

# Gravitational constants
G = 9.81  # Gravitational acceleration at sea level (m/s²)
G_MOON = 1.62  # Gravitational acceleration on Moon (m/s²)
G_MARS = 3.71  # Gravitational acceleration on Mars (m/s²)

# Air properties at sea level, 15°C
AIR_DENSITY = 1.225  # Air density (kg/m³)
AIR_VISCOSITY = 1.81e-5  # Dynamic viscosity of air (Pa·s)

# Common drag coefficients
DRAG_COEFFICIENTS = {
    'sphere': 0.47,
    'cube': 1.05,
    'streamlined': 0.04,
    'cylinder': 0.82,
    'flat_plate': 1.28,
    'bullet': 0.295,
    'baseball': 0.35,
    'golf_ball': 0.25,
}

# Default projectile properties
DEFAULT_MASS = 1.0  # kg
DEFAULT_RADIUS = 0.05  # m (5 cm radius)
DEFAULT_DRAG_COEF = DRAG_COEFFICIENTS['sphere']

def cross_sectional_area(radius: float) -> float:
    """Calculate cross-sectional area of a sphere."""
    return math.pi * radius ** 2

def reynolds_number(velocity: float, length: float, density: float = AIR_DENSITY, 
                    viscosity: float = AIR_VISCOSITY) -> float:
    """
    Calculate Reynolds number for flow characterization.
    Re < 1: Stokes flow (linear drag)
    Re > 1000: Turbulent flow (quadratic drag)
    """
    return (density * velocity * length) / viscosity