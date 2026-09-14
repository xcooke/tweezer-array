def calculate_trajectory(d, T, dt, num_steps):

    # minimum jerk quintic trajectory
    # s = t/T
    # x(t) = d * (10*s^3 - 15*s^4 + 6*s^5)

    trajectory = []
    amplitudes = []

    for step in range(num_steps + 1):
        t = step * dt
        s = t / T
        x = d * (10 * s**3 - 15 * s**4 + 6 * s**5)
        trajectory.append(x)
        amplitudes.append(0.2)

    return trajectory, amplitudes