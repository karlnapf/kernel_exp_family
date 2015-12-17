import matplotlib.pyplot as plt
import numpy as np

def visualise_array(Xs, Ys, A, samples=None):
    im = plt.imshow(A, origin='lower')
    im.set_extent([Xs.min(), Xs.max(), Ys.min(), Ys.max()])
    im.set_interpolation('nearest')
    im.set_cmap('gray')
    if samples is not None:
        plt.plot(samples[:, 0], samples[:, 1], 'bx')
    plt.ylim([Ys.min(), Ys.max()])
    plt.xlim([Xs.min(), Xs.max()])


def pdf_grid(Xs, Ys, est):
    D = np.zeros((len(Xs), len(Ys)))
    G = np.zeros(D.shape)
    
    # this is in-efficient, log_pdf_multiple on a 2d array is faster
    for i, x in enumerate(Xs):
        for j, y in enumerate(Ys):
            point = np.array([x, y])
            D[j, i] = est.log_pdf(point)
            G[j, i] = np.linalg.norm(est.grad(point))
    
    return D, G

def visualise_fit(est, X, Xs=None, Ys=None):
    # visualise found fit
    plt.figure()
    if Xs is None:
        Xs = np.linspace(-5, 5)
    
    if Ys is None:
        Ys = np.linspace(-5, 5)
    
    D, G = pdf_grid(Xs, Ys, est)
     
    plt.subplot(121)
    visualise_array(Xs, Ys, D, X)
    plt.title("log pdf")
     
    plt.subplot(122)
    visualise_array(Xs, Ys, G, X)
    plt.title("gradient norm")
    
    plt.tight_layout()