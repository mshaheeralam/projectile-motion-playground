from matplotlib import pyplot as plt
import numpy as np

def plot_trajectory(x, y, title='Projectile Motion', xlabel='Distance (m)', ylabel='Height (m)'):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Trajectory', color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.legend()
    plt.show()

def plot_multiple_trajectories(trajectories, title='Multiple Projectile Trajectories'):
    plt.figure(figsize=(10, 5))
    for x, y in trajectories:
        plt.plot(x, y)
    plt.title(title)
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.grid()
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.legend(['Trajectory {}'.format(i+1) for i in range(len(trajectories))])
    plt.show()