import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from classes.MCMC import MCMC, CustomMCMC, LogPosteriorFunction
from classes.Function import Function
import math
from classes.Plotter import Plotter
from classes.Estimator import Crude
import numpy as np
import matplotlib.pyplot as plt

def exercise_1():
    plotter = Plotter()

    A = 8
    m = 10
    n = 10000
    burn_in = n // 5
    unnormalized_density = Function(lambda x: A ** x / (math.gamma(x + 1)))
    mcmc = MCMC(
        unnormalized_density = unnormalized_density,
        m = m
    )
    c = sum([unnormalized_density.evaluate(x) for x in range(0, m + 1)])
    normalized_density = Function(lambda x: unnormalized_density.evaluate(x) / c)

    Ts = []
    for _ in range(1000):
        mcmc.run(n = n, burn_in = burn_in, method = 'mh_ordinary')
        mcmc.thin(gap = 10)
        # plotter.plot_function(normalized_density, title = 'Normalized Density', x_label = 'x', y_label = 'f(x)')
        # plotter.plot_histogram(mcmc.chain, bins = mcmc.m + 1, title = 'Histogram of the chain', x_label = 'i', y_label = 'Frequency')
        
        # fig, ax = plt.subplots(figsize=(10, 6))
        
        # counts, bins, patches = ax.hist(mcmc.chain, bins=range(m + 2), alpha=0.7, 
        #                                density=True, label='MCMC Histogram', 
        #                                align='left', rwidth=0.8)
        
        # x_vals = range(m + 1)
        # y_vals = [normalized_density.evaluate(x) for x in x_vals]
        # ax.plot(x_vals, y_vals, 'ro-', linewidth=2, markersize=8, 
        #         label='Theoretical Density', alpha=0.8)
        
        # ax.set_xlabel('x')
        # ax.set_ylabel('Probability/Density')
        # ax.set_title('Comparison: MCMC Chain vs Theoretical Density')
        # ax.legend()
        # ax.grid(True, alpha=0.3)
        # ax.set_xticks(range(m + 1))
        
        # plt.tight_layout()
        # plt.savefig(f'L5. Markov Chain/plots/Ex1_combined.png', dpi=300, bbox_inches='tight')
        # plt.show()
        # plt.close()

        T = 0
        for i in range(m + 1):
            observed = len([x for x in mcmc.chain if x == i])
            expected = normalized_density.evaluate(i) * len(mcmc.chain)
            T += ((observed - expected) ** 2) / expected

        Ts.append(T)

    plotter.plot_histogram(Ts, bins = 25, title = 'Histogram of T', x_label = 'T', y_label = 'Frequency')


def exercise_2():
    plotter = Plotter()

    A1, A2 = 4, 4
    m = 10
    n = 1000
    gap = n // 20
    burn_in = n // 5
    unnormalized_density = Function(lambda x, y: (A1 ** x) / (math.factorial(x)) * (A2 ** y) / (math.factorial(y)) if x + y <= m else 0)
    mcmc = MCMC(
        unnormalized_density = unnormalized_density,
        m = m
    )

    c = sum([unnormalized_density.evaluate([x, y]) for x in range(0, m + 1) for y in range(0, m + 1) if x + y <= m])
    normalized_density = Function(lambda x, y: unnormalized_density.evaluate([x, y]) / c)

    methods = [
        'mh_ordinary', 
        'mh_coordinate_wise', 
        'gibbs'
    ]
    Ts = []

    for method in methods:
        for _ in range(100):
            mcmc.run(n = n, burn_in = burn_in, method = method, condition = lambda x: x[0] + x[1] <= m)
            mcmc.thin(gap = gap)
            # plotter.plot_histogram(mcmc.chain, bins = mcmc.m, title = f'Histogram of the chain ({method})', x_label = 'i', y_label = 'j', savepath = None)

            T = 0
            for i in range(m + 1):
                for j in range(m + 1):
                    if i + j > m:
                        continue
                    observed = len([x for x in mcmc.chain if x == [i, j]])
                    expected = normalized_density.evaluate([i, j]) * len(mcmc.chain)
                    T += ((observed - expected) ** 2) / expected

            Ts.append(T)

        plotter.plot_histogram(Ts, bins = 25, title = 'Histogram of T', x_label = 'T', y_label = 'Frequency', savepath = None)

def exercise_3():
    sample_sizes = [10, 100, 1000]
    
    results = []
    for n in sample_sizes:
        result = run_bayesian_inference(n)
        results.append(result)
    
    valid_results = [r for r in results if r is not None]
    if valid_results:
        plot_results(valid_results)
        
        print(f"\n{'='*80}")
        print(f"SUMMARY COMPARISON")
        print(f"{'='*80}")
        print(f"{'n':>6} {'θ_true':>8} {'θ_est':>8} {'θ_std':>8} {'ψ_true':>8} {'ψ_est':>8} {'ψ_std':>8} {'Accept':>8} {'StepSize':>10}")
        print(f"{'-'*80}")
        
        for r in valid_results:
            print(f"{r['n']:>6} {r['true_theta']:>8.3f} {r['theta_mean']:>8.3f} {r['theta_std']:>8.3f} "
                  f"{r['true_psi']:>8.3f} {r['psi_mean']:>8.3f} {r['psi_std']:>8.3f} "
                  f"{r['diagnostics']['acceptance_rate']:>8.3f} {r['diagnostics']['final_step_size']:>10.6f}")

def run_bayesian_inference(n_observations):
    print(f"\n{'='*60}")
    print(f"Running Bayesian Inference with n={n_observations} observations")
    print(f"{'='*60}")
    
    rho = 0.5
    xi, gamma = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]])
    print(f"Generated: xi = {xi:.3f}, gamma = {gamma:.3f}")

    theta, psi = math.exp(xi), math.exp(gamma)
    print(f"True parameters: theta = {theta:.3f}, psi = {psi:.3f}")
    X = np.random.normal(theta, math.sqrt(psi), size=n_observations)

    def log_likelihood(theta_val, psi_val):
        if theta_val <= 0 or psi_val <= 0:
            return -np.inf
        log_lik = -n_observations/2 * math.log(2 * math.pi * psi_val)
        log_lik -= (1/(2 * psi_val)) * np.sum((X - theta_val) ** 2)
        return log_lik

    def log_prior(theta_val, psi_val):
        if theta_val <= 0 or psi_val <= 0:
            return -np.inf
        log_theta = math.log(theta_val)
        log_psi = math.log(psi_val)
        
        det_sigma = 1 - rho**2
        quadratic_form = (log_theta**2 - 2*rho*log_theta*log_psi + log_psi**2) / (2 * det_sigma)
        
        log_prior_val = -0.5 * math.log(2 * math.pi * det_sigma)
        log_prior_val -= quadratic_form
        log_prior_val -= math.log(theta_val * psi_val)
        
        return log_prior_val

    def log_posterior(theta_val, psi_val):
        return log_likelihood(theta_val, psi_val) + log_prior(theta_val, psi_val)

    def posterior_density(theta_val, psi_val):
        log_post = log_posterior(theta_val, psi_val)
        if log_post < -500:
            return 1e-300
        return math.exp(log_post)
    
    posterior_function = Function(lambda x, y: posterior_density(x, y))

    print(f"Running MCMC sampling...")
    
    n_iterations = min(10000, max(5000, n_observations * 10))
    burn_in = min(2000, n_iterations // 5)
    
    mcmc = CustomMCMC(
        unnormalized_density=posterior_function,
        log_posterior_func=log_posterior,
    )
    mcmc.run(n=n_iterations, burn_in=burn_in, domain='continuous', method='mh_ordinary', verbose=True)
    
    diagnostics = mcmc.get_diagnostics()
    print(f"\nMCMC Diagnostics:")
    print(f"  Chain length: {diagnostics['chain_length']}")
    print(f"  Final acceptance rate: {diagnostics['acceptance_rate']:.3f}")
    print(f"  Final step size: {diagnostics['final_step_size']:.6f}")
    
    if len(mcmc.chain) == 0:
        print("MCMC failed - no samples generated")
        return None
    
    theta_samples = [sample[0] for sample in mcmc.chain]
    psi_samples = [sample[1] for sample in mcmc.chain]

    theta_mean = np.mean(theta_samples)
    theta_std = np.std(theta_samples)
    psi_mean = np.mean(psi_samples)
    psi_std = np.std(psi_samples)
    
    print(f"\nPosterior Results:")
    print(f"  True theta: {theta:.3f}, Posterior: {theta_mean:.3f} ± {theta_std:.3f}")
    print(f"  True psi:   {psi:.3f}, Posterior: {psi_mean:.3f} ± {psi_std:.3f}")
    
    theta_coverage = abs(theta - theta_mean) <= 2 * theta_std
    psi_coverage = abs(psi - psi_mean) <= 2 * psi_std
    print(f"  Theta coverage (±2σ): {'✓' if theta_coverage else '✗'}")
    print(f"  Psi coverage (±2σ):   {'✓' if psi_coverage else '✗'}")

    return {
        'n': n_observations,
        'true_theta': theta,
        'true_psi': psi,
        'theta_samples': theta_samples,
        'psi_samples': psi_samples,
        'theta_mean': theta_mean,
        'theta_std': theta_std,
        'psi_mean': psi_mean,
        'psi_std': psi_std,
        'diagnostics': diagnostics,
        'theta_coverage': theta_coverage,
        'psi_coverage': psi_coverage
    }

def plot_results(results_list):
    fig, axes = plt.subplots(2, len(results_list), figsize=(5*len(results_list), 10))
    if len(results_list) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, results in enumerate(results_list):
        if results is None:
            continue
            
        n = results['n']
        theta_samples = results['theta_samples']
        psi_samples = results['psi_samples']
        true_theta = results['true_theta']
        true_psi = results['true_psi']
        
        axes[0, i].hist(theta_samples, bins=50, alpha=0.7, density=True, color='blue')
        axes[0, i].axvline(true_theta, color='red', linestyle='--', linewidth=2, label=f'True θ={true_theta:.2f}')
        axes[0, i].axvline(results['theta_mean'], color='orange', linestyle='-', linewidth=2, label=f'Est θ={results["theta_mean"]:.2f}')
        axes[0, i].set_title(f'Posterior θ (n={n})')
        axes[0, i].set_xlabel('θ')
        axes[0, i].set_ylabel('Density')
        axes[0, i].legend()
        
        axes[1, i].hist(psi_samples, bins=50, alpha=0.7, density=True, color='green')
        axes[1, i].axvline(true_psi, color='red', linestyle='--', linewidth=2, label=f'True ψ={true_psi:.2f}')
        axes[1, i].axvline(results['psi_mean'], color='orange', linestyle='-', linewidth=2, label=f'Est ψ={results["psi_mean"]:.2f}')
        axes[1, i].set_title(f'Posterior ψ (n={n})')
        axes[1, i].set_xlabel('ψ')
        axes[1, i].set_ylabel('Density')
        axes[1, i].legend()
    
    plt.tight_layout()
    # plt.savefig()
    plt.show()
    plt.close()

    fig, axes = plt.subplots(1, len(results_list), figsize=(5*len(results_list), 5))
    if len(results_list) == 1:
        axes = [axes]
        
    for i, results in enumerate(results_list):
        if results is None:
            continue
            
        n = results['n']
        theta_samples = results['theta_samples']
        psi_samples = results['psi_samples']
        true_theta = results['true_theta']
        true_psi = results['true_psi']
        
        subsample = slice(None, None, max(1, len(theta_samples) // 1000))
        axes[i].scatter(theta_samples[subsample], psi_samples[subsample], alpha=0.6, s=1, color='blue')
        axes[i].scatter([true_theta], [true_psi], color='red', s=100, marker='x', label='True values')
        axes[i].set_xlabel('θ')
        axes[i].set_ylabel('ψ')
        axes[i].set_title(f'Joint Posterior (n={n})')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('L5. Markov Chain/plots/joint_posterior.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    # exercise_1()
    # exercise_2()
    exercise_3()