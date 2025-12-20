
import numpy as np
from scipy.stats import norm

from aptapy.plotting import plt, setup_gca
from aptapy.models import Gaussian


def calculate_eta(r, sigma: float = 0.12) -> float:
    dist = norm(loc=r, scale=sigma)
    q1 = dist.cdf(0.5) - dist.cdf(-0.5)
    q2 = dist.cdf(1.5) - dist.cdf(0.5)
    eta = q2 / (q1 + q2)
    return eta

def calculate_r(eta: float, sigma: float = 0.12) -> float:
    r = 0.5 + norm.ppf(eta, scale=sigma)
    try:
        r[np.isinf(r)] = 0.
    except TypeError:
        pass
    return r

# r = np.linspace(0., 0.5, 25)
# eta = [calculate_eta(_r, sigma) for _r in r]


eta = np.linspace(0., 0.5, 500)
bbox_kwargs = dict(facecolor="white", edgecolor="none", pad=0.1, alpha=0.9)
label_kwargs = dict(fontsize="small", va='center', ha='center', bbox=bbox_kwargs)
for denom in 15, 10, 8, 6, 5, 4:
    sigma = 1.0 / denom
    plt.plot(eta, calculate_r(eta, sigma), color="black")
    x = 0.075
    y = calculate_r(x, sigma)
    plt.text(x, y, rf'$p/{denom}$', **label_kwargs)
plt.plot(eta, eta, color="black", ls="--")
setup_gca(xlabel=r'$\eta$', ylabel=r'$r/p$', xmin=0., xmax=0.5, ymin=0., ymax=0.5)

plt.savefig("gauss1d_recon.pdf")

# gauss = Gaussian()
# gauss.mu.set(mean)
# gauss.sigma.set(sigma)
# gauss.plot()

# q1 = gauss.integral(-0.5, 0.5)
# q2 = gauss.integral(0.5, 1.5)
# eta = q2 / (q1 + q2)
# print(q1, q2, eta)
# print(calculate_eta(mean, sigma))

# kwargs = dict(color="black", ls="--")
# plt.axvline(-0.5, **kwargs)
# plt.axvline(0.5, **kwargs)
# plt.axvline(1.5, **kwargs)

plt.show()


