import numpy as np
from scipy.stats import norm, chi2

ALPHA = 0.05

def mle_normal(sample):
    mu_hat = np.mean(sample)
    sigma_hat = np.std(sample, ddof=0)
    return float(mu_hat), float(sigma_hat)

def format_bound(x):
    if np.isneginf(x):
        return "-∞"
    if np.isposinf(x):
        return "+∞"
    return f"{x:.3f}"

def build_initial_bins(sample):
    n = len(sample)

    if n >= 50:
        k = int(np.ceil(1 + 3.3 * np.log10(n)))
    else:
        k = 10

    eps = 1e-9
    min_val = np.min(sample)
    max_val = np.max(sample)

    bins = np.linspace(min_val - eps, max_val + eps, k + 1)
    bins[0] = -np.inf
    bins[-1] = np.inf
    return bins


def merge_intervals(observed, expected, probs, bins):
    obs = observed.tolist()
    exp = expected.tolist()
    pr = probs.tolist()
    b = bins.tolist()

    while np.any(np.array(exp) < 5) and len(obs) > 2:
        idx = next(i for i, e in enumerate(exp) if e < 5)

        if idx == 0:
            obs[0] += obs[1]
            exp[0] += exp[1]
            pr[0] += pr[1]

            del obs[1]
            del exp[1]
            del pr[1]
            del b[1]
        else:
            obs[idx - 1] += obs[idx]
            exp[idx - 1] += exp[idx]
            pr[idx - 1] += pr[idx]

            del obs[idx]
            del exp[idx]
            del pr[idx]
            del b[idx]

    return np.array(obs), np.array(exp), np.array(pr), np.array(b)


def chi_square_normality_test(sample, alpha=ALPHA):
    n = len(sample)

    mu_hat, sigma_hat = mle_normal(sample)

    bins = build_initial_bins(sample)

    observed, _ = np.histogram(sample, bins=bins)

    probs = np.diff(norm.cdf(bins, loc=mu_hat, scale=sigma_hat))
    expected = n * probs

    observed, expected, probs, bins = merge_intervals(observed, expected, probs, bins)

    m = len(observed)
    df = m - 3

    if m < 3 or df <= 0:
        return {
            "n": n,
            "mu_hat": mu_hat,
            "sigma_hat": sigma_hat,
            "applicable": False,
            "reason": "После объединения интервалов критерий χ² неприменим."
        }

    diff = observed - expected
    contributions = diff ** 2 / expected
    chi2_obs = float(np.sum(contributions))
    chi2_crit = float(chi2.ppf(1 - alpha, df))
    p_value = float(1 - chi2.cdf(chi2_obs, df))

    return {
        "n": n,
        "mu_hat": mu_hat,
        "sigma_hat": sigma_hat,
        "bins": bins,
        "observed": observed,
        "probs": probs,
        "expected": expected,
        "diff": diff,
        "contributions": contributions,
        "df": df,
        "chi2_obs": chi2_obs,
        "chi2_crit": chi2_crit,
        "p_value": p_value,
        "accept_h0": chi2_obs < chi2_crit,
        "applicable": True
    }


def print_table(result):
    bins = result["bins"]
    observed = result["observed"]
    probs = result["probs"]
    expected = result["expected"]
    diff = result["diff"]
    contributions = result["contributions"]

    print(f"{'№':<3} {'Интервал':<24} {'n_i':>6} {'p_i':>10} {'n p_i':>10} {'n_i-n p_i':>12} {'(n_i-n p_i)^2/(n p_i)':>24}")
    for i in range(len(observed)):
        interval_str = f"({format_bound(bins[i])}; {format_bound(bins[i + 1])}]"
        print(
            f"{i + 1:<3} "
            f"{interval_str:<24} "
            f"{observed[i]:>6.0f} "
            f"{probs[i]:>10.4f} "
            f"{expected[i]:>10.2f} "
            f"{diff[i]:>12.2f} "
            f"{contributions[i]:>24.4f}"
        )

    print(f"\nχ² наблюдаемое = {result['chi2_obs']:.4f}")
    print(f"χ² критическое = {result['chi2_crit']:.4f} (df = {result['df']})")
    print(f"p-value = {result['p_value']:.4f}")
    print("Гипотеза H0", "принимается" if result["accept_h0"] else "отвергается")


def print_result(title, result):
    print(f"\n=== {title} ===")
    print(f"n = {result['n']}")
    print(f"mu_hat = {result['mu_hat']:.4f}")
    print(f"sigma_hat = {result['sigma_hat']:.4f}")

    if not result["applicable"]:
        print(result["reason"])
        return

    print(f"Число интервалов после объединения: {len(result['observed'])}")
    print()
    print_table(result)


def main():
    np.random.seed(13)

    sample_normal = np.random.normal(0, 1, 100)
    result_normal = chi_square_normality_test(sample_normal)
    print_result("Нормальное N(0,1), n=100", result_normal)

    a = np.sqrt(3)
    sample_uniform = np.random.uniform(-a, a, 20)
    result_uniform = chi_square_normality_test(sample_uniform)
    print_result("Равномерное U(-√3, √3), n=20", result_uniform)

    sample_laplace = np.random.laplace(0, 1 / np.sqrt(2), 20)
    result_laplace = chi_square_normality_test(sample_laplace)
    print_result("Лапласа L(0, 1/√2), n=20", result_laplace)

    # ДОПОЛНИТЕЛЬНОЕ ИССЛЕДОВАНИЕ ДЛЯ n = 100
    sample_uniform_100 = np.random.uniform(-a, a, 100)
    result_uniform_100 = chi_square_normality_test(sample_uniform_100)
    print_result("Равномерное U(-√3, √3), n=100", result_uniform_100)

    sample_laplace_100 = np.random.laplace(0, 1 / np.sqrt(2), 100)
    result_laplace_100 = chi_square_normality_test(sample_laplace_100)
    print_result("Лапласа L(0, 1/√2), n=100", result_laplace_100)


if __name__ == "__main__":
    main()