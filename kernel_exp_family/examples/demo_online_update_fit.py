from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
from kernel_exp_family.examples.demo_simple import ground_truth
from kernel_exp_family.examples.tools import pdf_grid, visualise_array
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    """
    Simple example of the on-line fitting abilities of the finite dimensional
    approximation.
    The model is initialised with a set of samples from a "wrong" density.
    Then, increasing amounts of "correct" data are added and the fits are
    visualised.
    """
    N = 50
    D = 2
    
    # fit model to samples from a wrong Gaussian, to see updates later
    X = np.random.randn(N, D) * 10
    
    # arbitrary choice of parameters here
    # note that m is set to N in order to call update_fit immediately,
    # as throws an error if called with less data
    sigma = 2
    lmbda = 0.001
    m = N
    est = KernelExpFiniteGaussian(sigma, lmbda, m, D)
    est.fit(X)

    # only for plotting
    all_data = [X]
    
    # plotting grid
    width = 6
    Xs = np.linspace(-width, width, 50)
    Ys = np.linspace(-width, width, 50)

    # plot ground truth
    plt.figure(figsize=(10, 10))
    fig_count = 1
    plt.subplot(3, 3, fig_count)
    _, G_true = pdf_grid(Xs, Ys, ground_truth())
    visualise_array(Xs, Ys, G_true, np.vstack(all_data))
    plt.title("Gradient norm, ground truth")
    
    # plot initial fit
    fig_count +=1
    plt.subplot(3, 3, fig_count)
    D, G = pdf_grid(Xs, Ys, est)
    visualise_array(Xs, Ys, G, np.vstack(all_data))
    plt.title("Gradient norm, initial fit, N=%d" % (est.n))
    plt.tight_layout()

    # online updates of the model
    for i in range(7):
        X = np.random.randn((i+1)*20, est.D)
        
        # API for updating estimator
        for x in X:
            est.update_fit(x)
        
        # only for plotting
        all_data.append(X)
            
        # visualise current fit
        fig_count += 1
        plt.subplot(3, 3, fig_count)
        D, G = pdf_grid(Xs, Ys, est)
        visualise_array(Xs, Ys, G, np.vstack(all_data))
        plt.title("Gradient norm, N=%d" % (est.n))
        plt.tight_layout()
        
    plt.show()
