import numpy as np
def print_table(results):
    print("Распределение | n | Характеристика | E ± σ")
    print("--------------------------------------------")
    for row in results:
        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")

rng = np.random.default_rng()
def generate_normal(n):
    return rng.normal(0, 1, n)

def generate_cauchy(n):
    return np.random.standard_cauchy(n)

def generate_laplace(n):
    return np.random.laplace(0, np.sqrt(1 / 2), n)

def generate_poisson(n):
    return np.random.poisson(5, n)

def generate_uniform(n):
    return np.random.uniform(-np.sqrt(3), np.sqrt(3), n)

def round_result(E, D):
    return round(E,1), round(np.sqrt(D),1)

def calculate(sample, n):
    x_mean = np.mean(sample)

    x_med = np.median(sample)

    xR = (min(sample) + max(sample)) / 2

    q1, q3 = np.percentile(sample, [25, 75])
    xQ = (q1 + q3) / 2

    sorted_sample = np.sort(sample)
    r = int(np.floor(0.1 * n))
    xTR = np.mean(sorted_sample[r:-r])

    return x_mean, x_med, xR, xQ, xTR

def main():
    sample_size = [10, 100, 1000]

    distribustions = {
        "Нормальное": generate_normal,
        "Коши": generate_cauchy,
        "Лапласа": generate_laplace,
        "Пуассона": generate_poisson,
        "Равномерное": generate_uniform
    }

    n_exp = 1000
    results = []

    for n in sample_size:
        for name, gen in distribustions.items():
            means = np.zeros(n_exp)
            meds = np.zeros(n_exp)
            xRs = np.zeros(n_exp)
            xQs = np.zeros(n_exp)
            xTRs = np.zeros(n_exp)

            for i in range(n_exp):
                means[i], meds[i], xRs[i], xQs[i], xTRs[i] = calculate(gen(n), n)

            stats = {
                "Выборочное среднее": means,
                "Медиана": meds,
                "zR": xRs,
                "zQ": xQs,
                "Усечённое среднее": xTRs
            }

            for stat_name, values in stats.items():
                E = np.mean(values)
                D = np.mean(values**2) - E**2
                E_r, D_r = round_result(E, D)
                results.append((name, n, stat_name, f"{E_r} ± {D_r}"))

    print_table(results)

if __name__ == "__main__":
    main()