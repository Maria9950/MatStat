import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(42)

def generate_samples(distribution, size):
    if distribution == "normal":
        return np.random.normal(0, 1, size)
    elif distribution == "cauchy":
        return np.random.standard_cauchy(size)
    elif distribution == "laplace":
        return np.random.laplace(0, 1/np.sqrt(2), size)
    elif distribution == "poisson":
        return np.random.poisson(10, size)
    elif distribution == "uniform":
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), size)

def plot_distribution(distribution, sizes):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, size in enumerate(sizes):
        samples = generate_samples(distribution, size)
        ax = axes[i]

        if distribution == "poisson":
            values, counts = np.unique(samples, return_counts=True)
            ax.bar(values, counts / size,
                   alpha=0.7,
                   color="#F4A261",      # нежно-оранжевый
                   edgecolor="black",
                   label="Гистограмма")

            k = np.arange(0, 25)
            ax.plot(k,
                    stats.poisson.pmf(k, 10),
                    marker='o',
                    linestyle='-',
                    color="black",
                    label="Теоретическая PMF")

            ax.set_ylabel("Масса вероятности")

        else:
            ax.hist(samples,
                    bins='fd',
                    density=True,
                    alpha=0.7,
                    color="#F4A261",      # нежно-оранжевый
                    edgecolor="black",
                    label="Гистограмма")

            x = np.linspace(-10, 10, 1000)

            if distribution == "normal":
                ax.plot(x, stats.norm.pdf(x, 0, 1),
                        color="black",
                        linewidth=2,
                        label="Теоретическая плотность")
                ax.set_xlim(-4, 4)

            elif distribution == "cauchy":
                ax.plot(x, stats.cauchy.pdf(x, 0, 1),
                        color="black",
                        linewidth=2,
                        label="Теоретическая плотность")
                ax.set_xlim(-10, 10)

            elif distribution == "laplace":
                ax.plot(x, stats.laplace.pdf(x, 0, 1/np.sqrt(2)),
                        color="black",
                        linewidth=2,
                        label="Теоретическая плотность")
                ax.set_xlim(-6, 6)

            elif distribution == "uniform":
                ax.plot(x, stats.uniform.pdf(x, -np.sqrt(3), 2*np.sqrt(3)),
                        color="black",
                        linewidth=2,
                        label="Теоретическая плотность")
                ax.set_xlim(-2.5, 2.5)

            ax.set_ylabel("Плотность вероятности")

        ax.set_xlabel("Значения случайной величины")
        ax.set_title(f"{distribution.capitalize()}, n = {size}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

sizes = [10, 100, 1000]
distributions = ["normal", "cauchy", "laplace", "poisson", "uniform"]

for dist in distributions:
    plot_distribution(dist, sizes)