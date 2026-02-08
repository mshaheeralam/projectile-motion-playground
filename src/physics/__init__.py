"""
Physics module for projectile motion simulation.
"""
from .constants import (
    G, G_MOON, G_MARS, AIR_DENSITY, AIR_VISCOSITY,
    DRAG_COEFFICIENTS, DEFAULT_MASS, DEFAULT_RADIUS, DEFAULT_DRAG_COEF,
    cross_sectional_area, reynolds_number
)
from .projectile import (
    ProjectileMotion, TrajectoryPoint,
    calculate_range, calculate_max_height, calculate_flight_time
)
from .air_drag import (
    ProjectileWithDrag, DragTrajectoryPoint, DragModel
)

__all__ = [
    # Constants
    'G', 'G_MOON', 'G_MARS', 'AIR_DENSITY', 'AIR_VISCOSITY',
    'DRAG_COEFFICIENTS', 'DEFAULT_MASS', 'DEFAULT_RADIUS', 'DEFAULT_DRAG_COEF',
    'cross_sectional_area', 'reynolds_number',
    # Projectile (no drag)
    'ProjectileMotion', 'TrajectoryPoint',
    'calculate_range', 'calculate_max_height', 'calculate_flight_time',
    # Projectile with drag
    'ProjectileWithDrag', 'DragTrajectoryPoint', 'DragModel',
]