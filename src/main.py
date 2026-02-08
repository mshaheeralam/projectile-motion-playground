import sys
from simulation.engine import SimulationEngine
from playground.app import start_playground

def main():
    simulation = SimulationEngine()
    simulation.run()
    start_playground()

if __name__ == "__main__":
    main()