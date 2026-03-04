import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

def generate_normal(n):
    return np.random.normal(0, 1, n)

def generate_cauchy(n):
    return np.random.standard_cauchy(n)

def generate_laplace(n):
    return np.random.laplace(0, 1 / np.sqrt(2), n)

def generate_poisson(n):
    return np.random.poisson(10, n)

def generate_uniform(n):
    return np.random.uniform(-np.sqrt(3), np.sqrt(3), n)

def outlier_fraction(sample):
    q1, q3 = np.quantile(sample, [0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = sample[(sample < lower) | (sample > upper)]
    return len(outliers) / len(sample)

def run_experiment():
    sample_sizes = [20, 100]
    distributions = {
        'Нормальное': generate_normal,
        'Коши': generate_cauchy,
        'Лапласа': generate_laplace,
        'Пуассона': generate_poisson,
        'Равномерное': generate_uniform
    }
    n_exp = 1000
    results = {}
    cauchy_combined = {}

    for dist_name, gen_func in distributions.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
        fig.suptitle(f'Распределение {dist_name}', fontsize=16, fontweight='bold')

        for idx, n in enumerate(sample_sizes):
            fracs = np.zeros(n_exp)
            samples_for_box = []
            for i in range(n_exp):
                sample = gen_func(n)
                fracs[i] = outlier_fraction(sample)
                if i < 5:
                    samples_for_box.append(sample)

            mean_frac = np.mean(fracs)
            std_frac = np.std(fracs)
            results[(dist_name, n)] = (mean_frac, std_frac)

            combined = np.concatenate(samples_for_box)

            if dist_name == 'Коши':
                cauchy_combined[n] = combined

            axes[idx].boxplot(combined, patch_artist=True,
                              boxprops=dict(facecolor='#F4A261', color='black'),
                              whiskerprops=dict(color='black'),
                              capprops=dict(color='black'),
                              medianprops=dict(color='darkred', linewidth=2),
                              flierprops=dict(marker='o', markerfacecolor='#999999', markersize=4, alpha=0.5))
            axes[idx].set_title(f'n = {n}\nСредняя доля выбросов: {mean_frac:.2%}  (± {std_frac:.2%})')
            axes[idx].set_ylabel('Значения')
            axes[idx].grid(axis='y', linestyle='--', alpha=0.7)

            # Убираем метки по оси X
            axes[idx].set_xticks([])

        plt.tight_layout()
        plt.show()

    if cauchy_combined:
        fig_cauchy, axes_cauchy = plt.subplots(1, 2, figsize=(14, 6))
        fig_cauchy.suptitle('Распределение Коши — центральная часть', fontsize=16, fontweight='bold')

        for idx, n in enumerate(sample_sizes):
            data = cauchy_combined[n]
            axes_cauchy[idx].boxplot(data, patch_artist=True,
                                     boxprops=dict(facecolor='#F4A261'),
                                     whiskerprops=dict(color='black'),
                                     medianprops=dict(color='darkred', linewidth=2))
            axes_cauchy[idx].set_title(f'n = {n}')
            axes_cauchy[idx].set_ylabel('Значения')
            axes_cauchy[idx].grid(axis='y', linestyle='--', alpha=0.7)

            # Ограничиваем ось Y для показа центральной части
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            axes_cauchy[idx].set_ylim(q1 - 3*iqr, q3 + 3*iqr)

            # Убираем метки по оси X
            axes_cauchy[idx].set_xticks([])

        plt.tight_layout()
        plt.show()

    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    print(f"{'Распределение':<15} {'n':<6} {'Средняя доля выбросов':<22} {'Стд. отклонение':<18}")
    print("-"*60)
    for (dist, n), (mean, std) in results.items():
        print(f"{dist:<15} {n:<6} {mean:.4%} ({mean:.2%})          {std:.4%} ({std:.2%})")
    print("="*60)

if __name__ == '__main__':
    run_experiment()