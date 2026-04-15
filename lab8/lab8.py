import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ALPHA = 0.05

def confidence_interval_mean(sample, alpha=ALPHA):

    n = len(sample)
    x_bar = np.mean(sample)
    s = np.std(sample, ddof=1)

    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_crit * s / np.sqrt(n)

    return float(x_bar - margin), float(x_bar + margin)

def confidence_interval_variance(sample, alpha=ALPHA):

    n = len(sample)
    s2 = np.var(sample, ddof=1)

    chi2_low = stats.chi2.ppf(alpha / 2, df=n - 1)
    chi2_high = stats.chi2.ppf(1 - alpha / 2, df=n - 1)

    left = (n - 1) * s2 / chi2_high
    right = (n - 1) * s2 / chi2_low

    return float(left), float(right)

def fisher_f_test(sample1, sample2, alpha=ALPHA):

    n1 = len(sample1)
    n2 = len(sample2)

    s1_sq = np.var(sample1, ddof=1)
    s2_sq = np.var(sample2, ddof=1)

    f_obs = s1_sq / s2_sq
    df1 = n1 - 1
    df2 = n2 - 1

    f_crit_low = stats.f.ppf(alpha / 2, df1, df2)
    f_crit_high = stats.f.ppf(1 - alpha / 2, df1, df2)

    p_left = stats.f.cdf(f_obs, df1, df2)
    p_right = 1 - p_left
    p_value = min(2 * min(p_left, p_right), 1.0)

    accept_h0 = f_crit_low <= f_obs <= f_crit_high

    return {
        "s1_sq": float(s1_sq),
        "s2_sq": float(s2_sq),
        "f_obs": float(f_obs),
        "f_crit_low": float(f_crit_low),
        "f_crit_high": float(f_crit_high),
        "df1": df1,
        "df2": df2,
        "p_value": float(p_value),
        "accept_h0": accept_h0
    }

def print_sample_info(name, sample, alpha=ALPHA):
    n = len(sample)
    x_bar = np.mean(sample)
    s2 = np.var(sample, ddof=1)

    ci_mean = confidence_interval_mean(sample, alpha)
    ci_var = confidence_interval_variance(sample, alpha)

    print(f"{name}")
    print(f"n = {n}")
    print(f"x̄ = {x_bar:.4f}")
    print(f"s² = {s2:.4f}")
    print(f"Доверительный интервал для m: ({ci_mean[0]:.4f}; {ci_mean[1]:.4f})")
    print(f"Доверительный интервал для σ²: ({ci_var[0]:.4f}; {ci_var[1]:.4f})")
    print()


def print_f_test_result(result):
    print("F-тест на равенство дисперсий")
    print(f"s1² = {result['s1_sq']:.4f}")
    print(f"s2² = {result['s2_sq']:.4f}")
    print(f"F_набл = {result['f_obs']:.4f}")
    print(f"Критический интервал: [{result['f_crit_low']:.4f}; {result['f_crit_high']:.4f}]")
    print(f"p-value = {result['p_value']:.4f}")

    if result["accept_h0"]:
        print("Вывод: нет оснований отвергать гипотезу H0 о равенстве дисперсий.")
    else:
        print("Вывод: гипотеза H0 о равенстве дисперсий отвергается.")
    print()


def plot_histogram(sample, title):
    plt.figure(figsize=(7, 4.5))
    plt.hist(sample, bins="fd", density=True, alpha=0.7, edgecolor="black", label="Выборка")

    x_min = min(-4, np.min(sample) - 0.5)
    x_max = max(4, np.max(sample) + 0.5)
    x = np.linspace(x_min, x_max, 1000)

    plt.plot(x, stats.norm.pdf(x, 0, 1), linewidth=2, label="Плотность N(0,1)")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("Плотность")
    plt.grid(True)
    plt.legend()


def plot_boxplots(sample1, sample2):
    plt.figure(figsize=(6, 4.5))
    plt.boxplot([sample1, sample2], labels=["n=20", "n=100"])
    plt.title("Сравнение выборок")
    plt.grid(True)


def main():
    np.random.seed(13)

    n1, n2 = 20, 100
    sample1 = np.random.normal(0, 1, n1)
    sample2 = np.random.normal(0, 1, n2)

    print_sample_info("Выборка из N(0,1), n = 20", sample1)
    print_sample_info("Выборка из N(0,1), n = 100", sample2)

    f_test_result = fisher_f_test(sample1, sample2)
    print_f_test_result(f_test_result)

    plot_histogram(sample1, "Гистограмма выборки n = 20")
    plot_histogram(sample2, "Гистограмма выборки n = 100")
    plot_boxplots(sample1, sample2)

    plt.show()


if __name__ == "__main__":
    main()