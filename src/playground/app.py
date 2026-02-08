"""
Projectile Motion Prediction Playground - Streamlit App
Interactive visualization comparing physics simulation vs ML prediction.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from physics.projectile import ProjectileMotion
from physics.air_drag import ProjectileWithDrag, DragModel
from physics.constants import G, DRAG_COEFFICIENTS, AIR_DENSITY
from simulation.engine import SimulationEngine, SimulationConfig, SolutionMethod
from simulation.data_generator import DatasetGenerator
from ml.model import (
    LinearRegressionModel, RidgeRegressionModel, 
    RandomForestModel, MLPModel
)


def setup_page():
    """Configure page settings."""
    st.set_page_config(
        page_title="Projectile Motion Playground",
        page_icon="🎯",
        layout="wide"
    )
    st.title("🎯 Projectile Motion Prediction Playground")
    st.markdown("""
    Compare physics simulations (analytical & numerical) with machine learning predictions.
    Explore how air drag affects trajectories and see when ML can match physics!
    """)


def sidebar_controls():
    """Create sidebar controls and return parameters."""
    st.sidebar.header("⚙️ Simulation Parameters")
    
    # Initial conditions
    st.sidebar.subheader("Launch Conditions")
    v0 = st.sidebar.slider("Initial Velocity (m/s)", 5.0, 100.0, 30.0, 1.0)
    angle = st.sidebar.slider("Launch Angle (°)", 5.0, 85.0, 45.0, 1.0)
    
    # Physical properties
    st.sidebar.subheader("Projectile Properties")
    mass = st.sidebar.slider("Mass (kg)", 0.1, 10.0, 1.0, 0.1)
    
    # Drag settings
    st.sidebar.subheader("Air Resistance")
    enable_drag = st.sidebar.checkbox("Enable Air Drag", value=True)
    
    drag_model = DragModel.NONE
    drag_coef = 0.47
    radius = 0.05
    
    if enable_drag:
        drag_type = st.sidebar.selectbox(
            "Drag Type",
            ["Quadratic (realistic)", "Linear (simplified)"]
        )
        drag_model = DragModel.QUADRATIC if "Quadratic" in drag_type else DragModel.LINEAR
        
        projectile_type = st.sidebar.selectbox(
            "Projectile Shape",
            list(DRAG_COEFFICIENTS.keys())
        )
        drag_coef = DRAG_COEFFICIENTS[projectile_type]
        radius = st.sidebar.slider("Radius (m)", 0.01, 0.2, 0.05, 0.01)
    
    # Numerical parameters
    st.sidebar.subheader("Numerical Settings")
    dt = st.sidebar.select_slider(
        "Time Step (s)",
        options=[0.1, 0.05, 0.01, 0.005, 0.001],
        value=0.01
    )
    
    return {
        'v0': v0,
        'angle': angle,
        'mass': mass,
        'drag_model': drag_model,
        'drag_coef': drag_coef,
        'radius': radius,
        'dt': dt,
        'enable_drag': enable_drag
    }


def run_physics_simulation(params):
    """Run physics simulation with given parameters."""
    results = {}
    
    if params['drag_model'] == DragModel.NONE:
        # No drag - can compare analytical vs numerical
        projectile = ProjectileMotion(
            v0=params['v0'],
            angle=params['angle']
        )
        
        results['analytical'] = {
            'trajectory': projectile.trajectory_analytical(num_points=200),
            'range': projectile.range_analytical(),
            'max_height': projectile.max_height_analytical(),
            'flight_time': projectile.time_of_flight_analytical()
        }
        
        results['euler'] = {
            'trajectory': projectile.trajectory_euler(dt=params['dt'])
        }
        
        results['rk4'] = {
            'trajectory': projectile.trajectory_rk4(dt=params['dt'])
        }
        
    else:
        # With drag - numerical only
        projectile = ProjectileWithDrag(
            v0=params['v0'],
            angle=params['angle'],
            mass=params['mass'],
            drag_coefficient=params['drag_coef'],
            radius=params['radius']
        )
        
        # Also run without drag for comparison
        projectile_no_drag = ProjectileMotion(
            v0=params['v0'],
            angle=params['angle']
        )
        
        results['no_drag'] = {
            'trajectory': projectile_no_drag.trajectory_analytical(num_points=200),
            'range': projectile_no_drag.range_analytical(),
            'max_height': projectile_no_drag.max_height_analytical(),
            'flight_time': projectile_no_drag.time_of_flight_analytical()
        }
        
        traj_drag = projectile.trajectory_rk4(dt=params['dt'], drag_model=params['drag_model'])
        results['with_drag'] = {
            'trajectory': traj_drag,
            'range': traj_drag[-1].x if traj_drag else 0,
            'max_height': max(p.y for p in traj_drag) if traj_drag else 0,
            'flight_time': traj_drag[-1].t if traj_drag else 0,
            'energy_loss': 100 * (1 - traj_drag[-1].energy / traj_drag[0].energy) if traj_drag else 0
        }
    
    return results


def plot_trajectories(results, params):
    """Plot trajectory comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if params['drag_model'] == DragModel.NONE:
        # Plot analytical vs numerical
        traj_a = results['analytical']['trajectory']
        traj_e = results['euler']['trajectory']
        traj_r = results['rk4']['trajectory']
        
        ax.plot([p.x for p in traj_a], [p.y for p in traj_a], 
                'b-', linewidth=2, label='Analytical (exact)')
        ax.plot([p.x for p in traj_e], [p.y for p in traj_e], 
                'r--', linewidth=1.5, label=f'Euler (dt={params["dt"]}s)')
        ax.plot([p.x for p in traj_r], [p.y for p in traj_r], 
                'g:', linewidth=2, label=f'RK4 (dt={params["dt"]}s)')
        
    else:
        # Plot with vs without drag
        traj_no = results['no_drag']['trajectory']
        traj_drag = results['with_drag']['trajectory']
        
        ax.plot([p.x for p in traj_no], [p.y for p in traj_no], 
                'b--', linewidth=2, label='No air resistance', alpha=0.7)
        ax.plot([p.x for p in traj_drag], [p.y for p in traj_drag], 
                'r-', linewidth=2, label=f'{params["drag_model"].value.capitalize()} drag')
        
        # Fill area to show drag effect
        ax.fill_between(
            [p.x for p in traj_drag], 
            [p.y for p in traj_drag], 
            alpha=0.2, color='red'
        )
    
    ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
    ax.set_ylabel('Height (m)', fontsize=12)
    ax.set_title('Projectile Trajectory', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.axhline(y=0, color='k', linewidth=0.5)
    
    return fig


def plot_energy(results, params):
    """Plot energy over time for drag simulation."""
    if params['drag_model'] == DragModel.NONE:
        return None
    
    traj = results['with_drag']['trajectory']
    if not traj:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    t = [p.t for p in traj]
    energy = [p.energy for p in traj]
    speed = [p.speed for p in traj]
    drag = [p.drag for p in traj]
    
    # Energy plot
    ax1.plot(t, energy, 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Kinetic Energy (J)')
    ax1.set_title('Energy Loss Due to Drag')
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(t, energy, alpha=0.3)
    
    # Speed and drag plot
    ax2.plot(t, speed, 'g-', linewidth=2, label='Speed (m/s)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, drag, 'r--', linewidth=2, label='Drag Force (N)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Speed (m/s)', color='g')
    ax2_twin.set_ylabel('Drag Force (N)', color='r')
    ax2.set_title('Speed and Drag vs Time')
    ax2.grid(True, alpha=0.3)
    
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    return fig


def show_metrics(results, params):
    """Display key metrics."""
    if params['drag_model'] == DragModel.NONE:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Range", f"{results['analytical']['range']:.2f} m")
        with col2:
            st.metric("Max Height", f"{results['analytical']['max_height']:.2f} m")
        with col3:
            st.metric("Flight Time", f"{results['analytical']['flight_time']:.2f} s")
        
        # Show numerical errors
        st.subheader("Numerical Method Errors")
        
        euler_traj = results['euler']['trajectory']
        rk4_traj = results['rk4']['trajectory']
        
        error_data = {
            'Method': ['Euler', 'RK4'],
            'Range Error (m)': [
                abs(euler_traj[-1].x - results['analytical']['range']) if euler_traj else 0,
                abs(rk4_traj[-1].x - results['analytical']['range']) if rk4_traj else 0
            ],
            'Height Error (m)': [
                abs(max(p.y for p in euler_traj) - results['analytical']['max_height']) if euler_traj else 0,
                abs(max(p.y for p in rk4_traj) - results['analytical']['max_height']) if rk4_traj else 0
            ]
        }
        st.dataframe(pd.DataFrame(error_data), hide_index=True)
        
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        no_drag = results['no_drag']
        with_drag = results['with_drag']
        
        with col1:
            delta = with_drag['range'] - no_drag['range']
            st.metric("Range", f"{with_drag['range']:.2f} m", 
                     f"{delta:.1f} m", delta_color="inverse")
        with col2:
            delta = with_drag['max_height'] - no_drag['max_height']
            st.metric("Max Height", f"{with_drag['max_height']:.2f} m",
                     f"{delta:.1f} m", delta_color="inverse")
        with col3:
            delta = with_drag['flight_time'] - no_drag['flight_time']
            st.metric("Flight Time", f"{with_drag['flight_time']:.2f} s",
                     f"{delta:.2f} s", delta_color="inverse")
        with col4:
            st.metric("Energy Lost", f"{with_drag['energy_loss']:.1f}%")


def ml_comparison_section():
    """Section for ML model comparison."""
    st.header("🤖 Machine Learning Comparison")
    
    st.markdown("""
    Train ML models to predict projectile outcomes and compare with physics simulation.
    The models learn from simulated data - see how well they generalize!
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider("Training Samples", 100, 5000, 1000, 100)
        include_drag = st.checkbox("Include Drag in Training Data", value=True)
    
    with col2:
        target = st.selectbox(
            "Prediction Target",
            ["Range", "Max Height", "Flight Time"]
        )
        models_to_train = st.multiselect(
            "Models to Compare",
            ["Linear Regression", "Ridge", "Random Forest", "MLP"],
            default=["Linear Regression", "Random Forest", "MLP"]
        )
    
    if st.button("🚀 Train & Compare Models"):
        with st.spinner("Generating training data..."):
            from simulation.data_generator import DatasetConfig
            
            config = DatasetConfig(
                num_samples=n_samples,
                include_no_drag=not include_drag,
                include_linear_drag=False,
                include_quadratic_drag=include_drag
            )
            generator = DatasetGenerator(config)
            df = generator.generate(show_progress=False)
        
        st.success(f"Generated {len(df)} samples")
        
        # Prepare data
        feature_cols = ['v0', 'angle', 'mass']
        
        # Column names depend on drag setting
        suffix = '_quad_drag' if include_drag else '_no_drag'
        target_map = {
            "Range": f"range{suffix}",
            "Max Height": f"max_height{suffix}", 
            "Flight Time": f"time_of_flight{suffix}"
        }
        target_col = target_map[target]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = []
        model_classes = {
            "Linear Regression": LinearRegressionModel,
            "Ridge": RidgeRegressionModel,
            "Random Forest": RandomForestModel,
            "MLP": MLPModel
        }
        
        progress_bar = st.progress(0)
        for i, model_name in enumerate(models_to_train):
            with st.spinner(f"Training {model_name}..."):
                model = model_classes[model_name]()
                model.fit(X_train, y_train, feature_names=feature_cols)
                metrics = model.evaluate(X_test, y_test)
                results.append({
                    'Model': model_name,
                    'RMSE': metrics.rmse,
                    'MAE': metrics.mae,
                    'R²': metrics.r2
                })
            progress_bar.progress((i + 1) / len(models_to_train))
        
        # Show results
        st.subheader("📊 Model Performance")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.highlight_min(['RMSE', 'MAE']).highlight_max(['R²']), 
                    hide_index=True)
        
        # Plot predictions vs actual
        st.subheader("Predictions vs Actual")
        best_model_name = results_df.loc[results_df['R²'].idxmax(), 'Model']
        best_model = model_classes[best_model_name]()
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.5, s=20)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
               'r--', linewidth=2, label='Perfect prediction')
        ax.set_xlabel(f'Actual {target}')
        ax.set_ylabel(f'Predicted {target}')
        ax.set_title(f'{best_model_name} - Best Model')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)


def step_size_analysis_section(params):
    """Analyze effect of numerical step size."""
    st.header("📐 Step Size Analysis")
    
    st.markdown("""
    See how the choice of time step affects numerical accuracy and computation time.
    Smaller steps = more accuracy but slower computation.
    """)
    
    if st.button("Run Step Size Analysis"):
        projectile = ProjectileWithDrag(
            v0=params['v0'],
            angle=params['angle'],
            mass=params['mass'],
            drag_coefficient=params['drag_coef'],
            radius=params['radius']
        )
        
        with st.spinner("Analyzing step sizes..."):
            analysis = projectile.analyze_step_size_effect(
                step_sizes=[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005],
                drag_model=params['drag_model'] if params['enable_drag'] else DragModel.NONE
            )
        
        results_df = pd.DataFrame(analysis['results'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(results_df[['step_size', 'range', 'range_error', 'num_steps']],
                        hide_index=True)
        
        with col2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            ax1.loglog(results_df['step_size'], results_df['range_error'], 'bo-')
            ax1.set_xlabel('Step Size (s)')
            ax1.set_ylabel('Range Error (m)')
            ax1.set_title('Accuracy vs Step Size')
            ax1.grid(True, alpha=0.3)
            
            ax2.loglog(results_df['step_size'], results_df['computation_time_ms'], 'ro-')
            ax2.set_xlabel('Step Size (s)')
            ax2.set_ylabel('Computation Time (ms)')
            ax2.set_title('Speed vs Step Size')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)


def main():
    """Main application entry point."""
    setup_page()
    
    # Get parameters from sidebar
    params = sidebar_controls()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Trajectory", "🤖 ML Comparison", "📐 Analysis"])
    
    with tab1:
        st.header("Physics Simulation")
        
        # Run simulation
        results = run_physics_simulation(params)
        
        # Show metrics
        show_metrics(results, params)
        
        # Plot trajectory
        st.subheader("Trajectory Plot")
        fig = plot_trajectories(results, params)
        st.pyplot(fig)
        
        # Energy plot (only for drag)
        if params['enable_drag']:
            st.subheader("Energy Analysis")
            energy_fig = plot_energy(results, params)
            if energy_fig:
                st.pyplot(energy_fig)
    
    with tab2:
        ml_comparison_section()
    
    with tab3:
        step_size_analysis_section(params)


if __name__ == "__main__":
    main()