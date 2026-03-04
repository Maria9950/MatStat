import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def generate_normal(n):
    return np.random.default_rng().normal(0, 1, n)

def generate_cauchy(n):
    return np.random.standard_cauchy(n)

def generate_laplace(n):
    return np.random.laplace(0, np.sqrt(1 / 2), n)

def generate_poisson(n):
    return np.random.poisson(10, n)

def generate_uniform(n):
    return np.random.uniform(-np.sqrt(3), np.sqrt(3), n)


def silverman_bandwidth(sample):
    n = len(sample)
    return 1.06 * np.std(sample) * n ** (-1 / 5)

def robust_silverman_bandwidth(sample):
    n = len(sample)
    iqr = np.quantile(sample, 0.75) - np.quantile(sample, 0.25)
    return 0.9 * min(np.std(sample), iqr / 1.34) * n ** (-1 / 5)


def gaussian_kde(x, sample, h):
    kde = np.zeros_like(x)
    for xi in sample:
        kde += np.exp(-0.5 * ((x - xi) / h) ** 2)
    return kde / (len(sample) * h * np.sqrt(2 * np.pi))

def plot_distribution(samples, name):
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    plt.style.use('seaborn-v0_8-whitegrid')
    fig.suptitle(f"Распределение {name}", fontsize=17, weight="bold")

    for j, sample in enumerate(samples):
        n = len(sample)

        if name == "Poisson":
            low, high = 6, 14
            x = np.arange(low, high + 1)
        else:
            low, high = -4, 4
            x = np.linspace(low, high, 1200)

        h = robust_silverman_bandwidth(sample) if name == "Cauchy" else silverman_bandwidth(sample)


        ax = axes[0, j]
        ax.grid(alpha=0.6)

        if name == "Poisson":
            values, counts = np.unique(sample, return_counts=True)
            ax.bar(values, counts / n,
                   color="navajowhite", alpha=0.7, label="Гистограмма")

            ax.step(x, st.poisson.pmf(x, 10), where='mid',
                    linestyle="--", color="deepskyblue",
                    linewidth=2, label="Теоретическая пл.")
        else:
            ax.hist(sample, bins="fd", range=(low, high), density=True,
                    color="navajowhite", alpha=0.7, label="Гистограмма")

            pdf = {
                "Normal": st.norm.pdf(x),
                "Laplace": st.laplace.pdf(x, 0, np.sqrt(1 / 2)),
                "Uniform": st.uniform.pdf(x, -np.sqrt(3), 2 * np.sqrt(3)),
                "Cauchy": st.cauchy.pdf(x)
            }

            ax.plot(x, pdf[name], linestyle="--",
                    linewidth=2.5, color="darkorange",
                    label="Теоретическая пл.")

            ax.plot(x, gaussian_kde(x, sample, h),
                    linewidth=2, color="peru",
                    label="Ядерная оценка пл.")

        ax.set_title(f"n = {n}")
        ax.set_xlim(low, high)
        ax.set_xlabel("Значение случайной величины")
        ax.set_ylabel("Плотность вероятности")
        ax.legend(fontsize=9, frameon=False)

        ax = axes[1, j]
        ax.grid(alpha=0.6)

        xs = np.sort(sample)
        ys = np.arange(1, n + 1) / n
        ax.plot(xs, ys, drawstyle="steps-post",
                color="saddlebrown", linewidth=2,
                label="ЭФР")

        cdf = {
            "Normal": st.norm.cdf(x),
            "Laplace": st.laplace.cdf(x, 0, np.sqrt(1 / 2)),
            "Uniform": st.uniform.cdf(x, -np.sqrt(3), 2 * np.sqrt(3)),
            "Cauchy": st.cauchy.cdf(x),
            "Poisson": st.poisson.cdf(x, 10)
        }

        if name == "Poisson":
            ax.step(x, cdf[name], where='post',
                    linestyle="--", linewidth=2.5,
                    color="darkorange", label="Теоретическая ФР")
        else:
            ax.plot(x, cdf[name],
                    linestyle="--", linewidth=2.5,
                    color="darkorange", label="Теоретическая ФР")

        ax.set_xlim(low, high)
        ax.set_xlabel("Значение случайной величины")
        ax.set_ylabel("F(x)")
        ax.legend(fontsize=9, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

def main():
    sizes = [20, 60, 100]
    distributions = {
        "Normal": generate_normal,
        "Cauchy": generate_cauchy,
        "Laplace": generate_laplace,
        "Poisson": generate_poisson,
        "Uniform": generate_uniform
    }

    for name, gen in distributions.items():
        plot_distribution([gen(n) for n in sizes], name)

    plt.show()

if __name__ == "__main__":
    main()