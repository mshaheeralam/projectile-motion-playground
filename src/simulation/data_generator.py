def generate_dataset(initial_conditions, num_samples):
    import numpy as np
    import pandas as pd
    from physics.projectile import Projectile
    from physics.air_drag import AirDrag

    data = []

    for _ in range(num_samples):
        angle = np.random.uniform(0, 90)  # Random launch angle between 0 and 90 degrees
        velocity = np.random.uniform(10, 100)  # Random initial velocity between 10 and 100 m/s
        drag_coefficient = np.random.uniform(0.1, 1.0)  # Random drag coefficient

        projectile = Projectile(velocity, angle)
        air_drag = AirDrag(drag_coefficient)

        # Calculate trajectory with and without air drag
        trajectory_no_drag = projectile.calculate_trajectory()
        trajectory_with_drag = air_drag.calculate_trajectory(projectile)

        data.append({
            'angle': angle,
            'velocity': velocity,
            'drag_coefficient': drag_coefficient,
            'trajectory_no_drag': trajectory_no_drag,
            'trajectory_with_drag': trajectory_with_drag
        })

    return pd.DataFrame(data)