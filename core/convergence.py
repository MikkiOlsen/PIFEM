import numpy as np
import matplotlib.pyplot as plt
import time
import gc

def run_convergence_study(problem_func, resolutions, title="Mesh Convergence Study", scale=1000, metric_label="Max Deflection [mm]", display_cpu_time = True, digits=7):
    results = []
    n_elements = []
    times = []

    print("-" * 40)
    print(f"{'Mesh (nx x ny)':<20} | {metric_label:<20} | {'Time (s)':<10}")
    print("-" * 40)

    for nx, ny in resolutions:
        start_time = time.time()

        metric = problem_func(nx, ny)

        end_time = time.time()

        results.append(metric * scale)
        n_elements.append(nx * ny)
        times.append(end_time - start_time)

        print(f"{f'{nx} x {ny}':<20} | {metric*scale:<20.{digits}f} | {end_time - start_time:<10.2f}")
        gc.collect()


    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Number of Elements')
    ax1.set_ylabel(metric_label, color='black')
    ax1.plot(n_elements, results, 'o-', color='black', linewidth=2,
             markerfacecolor='white', markeredgecolor='black', label='Deflection')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    if display_cpu_time:
        ax2 = ax1.twinx()
        color = 'black'
        ax2.set_ylabel('CPU Time [s]', color=color)
        ax2.plot(n_elements, times, 'o-', color=color,
                markerfacecolor=color, label='Time')
        #ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor=color)
        lines2, labels2 = ax2.get_legend_handles_labels()

    lines1, labels1 = ax1.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best') if display_cpu_time else ax1.legend(lines1, labels1, loc='best')

    plt.title(title)
    fig.tight_layout()

    if len(results) > 1:
        final_change = abs((results[-1] - results[-2]) / results[-1]) * 100
        print("-" * 40)
        print(f"Final Convergence Step Change: {final_change:.4f}%")

    plt.show()

    return n_elements, results, times
