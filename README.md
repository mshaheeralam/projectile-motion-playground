# 🎯 Projectile Motion Prediction Playground

A physics simulation and machine learning project that predicts projectile trajectories, comparing classical physics solutions with ML predictions.

## Overview

This project combines physics and computer science to:
1. **Simulate projectile motion** using analytical and numerical methods
2. **Model air resistance** using linear and quadratic drag
3. **Train ML models** to predict landing positions and compare with physics
4. **Visualize everything** in an interactive Streamlit playground

## Features

### Physics Engine
- **Analytical solutions** (exact, closed-form) for gravity-only motion
- **Numerical integration**: Euler and RK4 methods
- **Air drag models**: Linear (Stokes) and quadratic (turbulent) drag
- **Step-size analysis**: Accuracy vs performance trade-offs
- **Energy tracking**: Kinetic energy loss due to drag

### Machine Learning
- **Baseline models**: Linear Regression, Ridge Regression
- **Ensemble methods**: Random Forest
- **Neural networks**: MLP Regressor
- **Model comparison**: Side-by-side accuracy metrics

### Interactive Playground (Streamlit)
- Real-time trajectory visualization
- Toggle between physics methods and drag models
- Train and compare ML models on generated data
- Adjustable parameters: velocity, angle, mass, drag coefficient

## Project Structure
```
projectile-motion-playground/
├── src/
│   ├── physics/          # Physics calculations
│   │   ├── constants.py  # Physical constants (g, drag coefficients)
│   │   ├── projectile.py # Motion without drag (analytical + numerical)
│   │   └── air_drag.py   # Motion with drag (numerical only)
│   ├── simulation/       # Simulation engine
│   │   ├── engine.py     # Unified simulation interface
│   │   └── data_generator.py  # Generate training data
│   ├── ml/               # Machine learning
│   │   ├── model.py      # Model classes (Linear, RF, MLP)
│   │   ├── training.py   # Training utilities
│   │   └── prediction.py # Prediction interface
│   ├── playground/       # Streamlit app
│   │   └── app.py        # Interactive UI
│   └── main.py           # CLI entry point
├── tests/                # Unit tests
├── data/                 # Training data, samples
├── models/               # Saved models
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/projectile-motion-playground.git
cd projectile-motion-playground

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Interactive Playground (Recommended)
```bash
streamlit run src/playground/app.py
```

### Quick Physics Demo
```python
from src.physics.projectile import ProjectileMotion

# Create projectile: 50 m/s at 45 degrees
proj = ProjectileMotion(v0=50, angle=45)

# Get analytical results
print(f"Range: {proj.range_analytical():.2f} m")
print(f"Max Height: {proj.max_height_analytical():.2f} m")
print(f"Flight Time: {proj.time_of_flight_analytical():.2f} s")

# Compare numerical methods
comparison = proj.compare_methods(dt=0.01)
print(f"RK4 Error: {comparison['rk4']['range_error']:.4f} m")
```

### With Air Drag
```python
from src.physics.air_drag import ProjectileWithDrag, DragModel

# Baseball with quadratic drag
proj = ProjectileWithDrag(v0=40, angle=35, mass=0.145,
                          drag_coefficient=0.35, radius=0.037)

# Compare drag effects
results = proj.compare_drag_models(dt=0.001)
print(f"Range (no drag): {results['no_drag']['range']:.1f} m")
print(f"Range (quadratic): {results['quadratic_drag']['range']:.1f} m")
```

### Run Tests
```bash
python -m pytest tests/ -v
```

## Physics Background

### Equations of Motion (No Drag)
- $x(t) = x_0 + v_0 \cos(\theta) \cdot t$
- $y(t) = y_0 + v_0 \sin(\theta) \cdot t - \frac{1}{2}gt^2$

### Quadratic Drag Force
- $\vec{F}_{drag} = -\frac{1}{2} \rho C_d A |\vec{v}|^2 \hat{v}$

Where:
- $\rho$ = air density (1.225 kg/m³)
- $C_d$ = drag coefficient
- $A$ = cross-sectional area
- $\vec{v}$ = velocity vector

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.