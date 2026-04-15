import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


TRUE_A = 2.0
TRUE_B = 2.0


def least_squares_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mx = np.mean(x)
    my = np.mean(y)
    mx2 = np.mean(x * x)
    mxy = np.mean(x * y)

    b = (mxy - mx * my) / (mx2 - mx * mx)
    a = my - b * mx

    return float(a), float(b)


def least_modules_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n = len(x)

    c = np.hstack([np.ones(n), 0.0, 0.0])

    A_ub = np.zeros((2 * n, n + 2))
    b_ub = np.zeros(2 * n)

    for i in range(n):
        A_ub[i, i] = -1
        A_ub[i, n] = -1
        A_ub[i, n + 1] = -x[i]
        b_ub[i] = -y[i]

        A_ub[n + i, i] = -1
        A_ub[n + i, n] = 1
        A_ub[n + i, n + 1] = x[i]
        b_ub[n + i] = y[i]

    bounds = [(0, None)] * n + [(None, None), (None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError("Не удалось найти решение методом наименьших модулей.")

    a = result.x[n]
    b = result.x[n + 1]

    return float(a), float(b)


def abs_error(true_value: float, approx_value: float) -> float:
    return abs(true_value - approx_value)


def rel_error(true_value: float, approx_value: float) -> float:
    return abs(true_value - approx_value) / abs(true_value) * 100


def print_results_table(title: str, results: dict[str, tuple[float, float]]) -> None:
    print(f"\n{title}")
    print(
        f"{'Метод':<6} {'a':>9} {'Δa':>9} {'δa, %':>9} "
        f"{'b':>9} {'Δb':>9} {'δb, %':>9}"
    )

    for method_name, (a_hat, b_hat) in results.items():
        delta_a = abs_error(TRUE_A, a_hat)
        delta_b = abs_error(TRUE_B, b_hat)
        rel_a = rel_error(TRUE_A, a_hat)
        rel_b = rel_error(TRUE_B, b_hat)

        print(
            f"{method_name:<6} "
            f"{a_hat:9.4f} {delta_a:9.4f} {rel_a:9.2f} "
            f"{b_hat:9.4f} {delta_b:9.4f} {rel_b:9.2f}"
        )


def plot_dataset(ax, x: np.ndarray, y: np.ndarray, title: str) -> None:
    a_lsq, b_lsq = least_squares_fit(x, y)
    a_lad, b_lad = least_modules_fit(x, y)

    x_line = np.linspace(np.min(x), np.max(x), 300)

    ax.scatter(x, y, color='orange', s=35, label='Данные')
    ax.plot(x_line, TRUE_A + TRUE_B * x_line, color='green', linewidth=2, label='Истинная прямая')
    ax.plot(x_line, a_lsq + b_lsq * x_line, '--', linewidth=2, label='МНК')
    ax.plot(x_line, a_lad + b_lad * x_line, '-', linewidth=2, label='МНМ')

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()


def main() -> None:
    np.random.seed(5)

    x = np.arange(-1.8, 2.0 + 1e-12, 0.2)
    n = len(x)

    eps = np.random.normal(0, 1, n)
    y = TRUE_A + TRUE_B * x + eps

    y_outliers = y.copy()
    y_outliers[0] += 10
    y_outliers[-1] -= 10

    results_clean = {
        "МНК": least_squares_fit(x, y),
        "МНМ": least_modules_fit(x, y)
    }

    results_outliers = {
        "МНК": least_squares_fit(x, y_outliers),
        "МНМ": least_modules_fit(x, y_outliers)
    }

    print_results_table("Без выбросов", results_clean)
    print_results_table("С выбросами", results_outliers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_dataset(axes[0], x, y, "Регрессия без выбросов")
    plot_dataset(axes[1], x, y_outliers, "Регрессия с выбросами")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()