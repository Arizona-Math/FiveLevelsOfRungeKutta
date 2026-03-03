import math


def f(t: float, y: float) -> float:
    # Example ODE: dy/dt = t + y
    return t + y


def y_exact(t: float, y0: float) -> float:
    # Exact solution for y' = t + y with y(0)=y0:
    # y(t) = e^t*y0 + e^t - t - 1
    et = math.exp(t)
    return et * y0 + et - t - 1.0


def rk4_kernel(y_results, initial_conditions, h: float, steps: int, N: int) -> None:
    for idx in range(N):
        y = float(initial_conditions[idx])
        t = 0.0

        for _ in range(steps):
            k1 = h * f(t, y)
            k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
            k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
            k4 = h * f(t + h, y + k3)

            y = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            t += h

        y_results[idx] = y


def main():
    N = 10_000_000  # ten million (warning: Python will be slow + memory heavy)

    h = 0.1
    n_steps = int(math.floor(1.0 / h))  # same as C++

    # Allocate arrays (lists)
    y = [0.0] * N
    y0 = [0.0] * N

    # Initialize initial conditions: y0[i] = i
    for i in range(N):
        y0[i] = float(i)

    rk4_kernel(y, y0, h, n_steps, N)

    t_final = 1.0
    for i in range(100):
        yi_exact = y_exact(t_final, y0[i])
        err = yi_exact - y[i]
        print(f"i={i}, y0={y0[i]:g}, yi={y[i]:g}, yi_exact={yi_exact:g}, err={err:g}")


if __name__ == "__main__":
    main()
