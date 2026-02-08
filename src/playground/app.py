from streamlit import st
from simulation.engine import SimulationEngine
from physics.projectile import Projectile
from physics.air_drag import AirDrag
from ml.training import train_models
from ml.prediction import make_predictions

def main():
    st.title("Projectile Motion Prediction Playground")
    
    st.sidebar.header("Simulation Parameters")
    initial_velocity = st.sidebar.number_input("Initial Velocity (m/s)", min_value=0.0, value=10.0)
    launch_angle = st.sidebar.number_input("Launch Angle (degrees)", min_value=0.0, max_value=90.0, value=45.0)
    air_drag_enabled = st.sidebar.checkbox("Enable Air Drag", value=True)

    projectile = Projectile(initial_velocity, launch_angle)
    air_drag = AirDrag() if air_drag_enabled else None

    if st.button("Run Simulation"):
        engine = SimulationEngine(projectile, air_drag)
        trajectory_data = engine.run_simulation()
        
        st.line_chart(trajectory_data)

    if st.button("Train Models"):
        train_models()
        st.success("Models trained successfully!")

    if st.button("Make Predictions"):
        predictions = make_predictions()
        st.write(predictions)

if __name__ == "__main__":
    main()