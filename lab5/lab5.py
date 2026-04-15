import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


rho_mix = -0.7514

def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    res = spearmanr(x, y)
    return float(res.correlation)


def quadrant_corr(x: np.ndarray, y: np.ndarray) -> float:
    mx = np.median(x)
    my = np.median(y)

    sx = np.sign(x - mx)
    sy = np.sign(y - my)

    return float(np.mean(sx * sy))

def bivariate_normal(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    mean = np.array([0.0, 0.0])
    cov = np.array([
        [1.0, rho],
        [rho, 1.0]
    ])
    return rng.multivariate_normal(mean, cov, n)


def bivariate_mix(n: int, rng: np.random.Generator) -> np.ndarray:
    mean = np.array([0.0, 0.0])
    p1 = 0.9

    labels = rng.random(n) < p1
    sample = np.empty((n, 2), dtype=float)

    cov1 = np.array([
        [1.0, 0.9],
        [0.9, 1.0]
    ])
    cov2 = np.array([
        [100.0, -90.0],
        [-90.0, 100.0]
    ])

    n1 = int(labels.sum())
    n2 = n - n1

    if n1 > 0:
        sample[labels] = rng.multivariate_normal(mean, cov1, n1)
    if n2 > 0:
        sample[~labels] = rng.multivariate_normal(mean, cov2, n2)

    return sample

def ellipse_points(mx: float, my: float, sx: float, sy: float,
                   rho: float, C: float = 5.0, num: int = 400) -> tuple[np.ndarray, np.ndarray]:

    numerator = 2 * rho * sx * sy
    denominator = sx**2 - sy**2

    alpha = 0.5 * np.arctan2(numerator, denominator)

    term = np.sqrt((sx**2 - sy**2)**2 + 4 * (rho * sx * sy)**2)
    lambda1 = 0.5 * (sx**2 + sy**2 + term)
    lambda2 = 0.5 * (sx**2 + sy**2 - term)

    a = np.sqrt(C * lambda1)
    b = np.sqrt(C * lambda2)

    t = np.linspace(0, 2 * np.pi, num)
    xi = a * np.cos(t)
    eta = b * np.sin(t)

    u = xi * np.cos(alpha) - eta * np.sin(alpha)
    v = xi * np.sin(alpha) + eta * np.cos(alpha)

    x = mx + u
    y = my + v

    return x, y


def plot_scatter_with_ellipse(data: np.ndarray, rho: float | None = None,
                              C: float = 5.0, title: str = "") -> None:
    x = data[:, 0]
    y = data[:, 1]

    mx = float(np.mean(x))
    my = float(np.mean(y))
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))

    if rho is None:
        rho = float(np.corrcoef(x, y)[0, 1])

    ex, ey = ellipse_points(mx, my, sx, sy, rho, C)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=15, alpha=0.6, color='green', label="Выборка")
    plt.plot(ex, ey, linewidth=2, color='green', label="Эллипс")
    plt.scatter(mx, my, marker='x', s=100, color='black', label="Центр")

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

def main() -> None:
    sample_size = [20, 60, 100]
    rhos = [0, 0.5, 0.9]
    n_exp = 1000

    rng = np.random.default_rng(5)

    print("Двумерное нормальное распределение")
    print()

    for rho in rhos:
        for n in sample_size:
            pearsons = np.empty(n_exp, dtype=float)
            spearmans = np.empty(n_exp, dtype=float)
            quadrants = np.empty(n_exp, dtype=float)

            first_sample = None

            for i in range(n_exp):
                sample = bivariate_normal(n, rho, rng)

                if i == 0:
                    first_sample = sample.copy()

                x = sample[:, 0]
                y = sample[:, 1]

                pearsons[i] = pearson_corr(x, y)
                spearmans[i] = spearman_corr(x, y)
                quadrants[i] = quadrant_corr(x, y)

            pearson_m = float(np.mean(pearsons))
            pearson_d = float(np.nanvar(pearsons, ddof=1))

            spearman_m = float(np.mean(spearmans))
            spearman_d = float(np.nanvar(spearmans, ddof=1))

            quadrant_m = float(np.mean(quadrants))
            quadrant_d = float(np.nanvar(quadrants, ddof=1))

            print(f"sample size = {n}, rho = {rho}")
            print(f"pearson_m  = {pearson_m:.4f}, pearson_d  = {pearson_d:.4f}, |rho - pear|  = {abs(pearson_m - rho):.4f}")
            print(f"spearman_m = {spearman_m:.4f}, spearman_d = {spearman_d:.4f}, |rho - spear| = {abs(spearman_m - rho):.4f}")
            print(f"quadrant_m = {quadrant_m:.4f}, quadrant_d = {quadrant_d:.4f}, |rho - quad|  = {abs(quadrant_m - rho):.4f}")
            print()

            plot_scatter_with_ellipse(
                first_sample,
                rho=rho,
                title=f"Двумерное нормальное распределение, ρ = {rho}, n = {n}"
            )

    print("Двумерная смесь")
    print()

    for n in sample_size:
        pearsons = np.empty(n_exp, dtype=float)
        spearmans = np.empty(n_exp, dtype=float)
        quadrants = np.empty(n_exp, dtype=float)

        first_sample = None

        for i in range(n_exp):
            sample = bivariate_mix(n, rng)

            if i == 0:
                first_sample = sample.copy()

            x = sample[:, 0]
            y = sample[:, 1]

            pearsons[i] = pearson_corr(x, y)
            spearmans[i] = spearman_corr(x, y)
            quadrants[i] = quadrant_corr(x, y)

        pearson_m = float(np.mean(pearsons))
        pearson_d = float(np.nanvar(pearsons, ddof=1))

        spearman_m = float(np.mean(spearmans))
        spearman_d = float(np.nanvar(spearmans, ddof=1))

        quadrant_m = float(np.mean(quadrants))
        quadrant_d = float(np.nanvar(quadrants, ddof=1))

        print(f"sample size = {n}")
        print(f"pearson_m  = {pearson_m:.4f}, pearson_d  = {pearson_d:.4f}, |rho - pear|  = {abs(pearson_m - rho_mix):.4f}")
        print(f"spearman_m = {spearman_m:.4f}, spearman_d = {spearman_d:.4f}, |rho - spear| = {abs(spearman_m - rho_mix):.4f}")
        print(f"quadrant_m = {quadrant_m:.4f}, quadrant_d = {quadrant_d:.4f}, |rho - quad|  = {abs(quadrant_m - rho_mix):.4f}")
        print()

        plot_scatter_with_ellipse(
            first_sample,
            title=f"Двумерная смесь, n = {n}"
        )

    plt.show()


if __name__ == "__main__":
    main()