
import numpy as np

# The first function can be used to verify the value calculated in the second function

def eu_option(opt_type, mu, r, sig, T, K, s0):
    '''function to calculate options price with Black-Scholes formula.
    Args:
        opt_type: `call` or `put`.
        r: 
        sig: volatility.
        T: time to maturity.
        K: strike price.
        s0:initial price.
    Returns:
        European option price.
    '''
    assert opt_type in ['call', 'put'], 'Wrong argument '+opt_type
    d1 = (np.log(s0 / K) + (mu + sig ** 2 / 2) * T) / sig / T ** 0.5
    d2 = d1 - sig * T ** 0.5
    if opt_type == 'put':
        return (stats.norm.cdf(-d2) * K - stats.norm.cdf(-d1) * np.exp(mu * T) * s0) * np.exp(-r * T)
    else:
        return (np.exp(mu*T)*stats.norm.cdf(d1) * s0 - stats.norm.cdf(d2) * K) * np.exp(-r * T)

def us_option(opt_type, mu, r, sig, T, K, s0, theta=0.5, xstep=50, tstep=50, half=2):
    """function to solve PDE with finite difference method
    The default method is Crank-Nicolson.
    """
    assert opt_type in ['eu_call', 'eu_put', 'us_call', 'us_put']
    from scipy.sparse import diags
    xmin, xmax = np.log(s0 / K, dtype=np.float32) - half, np.log(s0 / K, dtype=np.float32) + half
    tau, deltax = sig ** 2 * T / 2 / tstep, half / xstep
    lam, q, qq = tau / deltax ** 2, 2 * r / sig ** 2, 2 * mu / sig ** 2
    A = diags([[1 + 2 * lam * theta], [-lam * theta], [-theta * lam]], [0, -1, 1], shape=(xstep * 2 + 1, xstep * 2 + 1)).toarray()
    B = diags([[1 - 2 * lam * (1 - theta)], [lam * (1 - theta)], [lam * (1 - theta)]], [0, -1, 1], shape=(xstep * 2 + 1, xstep * 2 + 1)).toarray()
    x = np.expand_dims(np.linspace(xmin, xmax, 2 * xstep + 1, dtype=np.float32), axis=1)
    d = np.zeros_like(x)
    if opt_type in ['eu_call', 'us_call']:
        w = np.maximum(np.exp(x / 2 * (qq + 1)) - np.exp(x / 2 * (qq - 1)), 0, dtype=np.float32)
        r1 = lambda t: 0.0
        r2 = lambda t: np.exp((qq + 1)/2 * xmax + (qq + 1)** 2 / 4 * t, dtype=np.float32)
    else:
        w = np.maximum(np.exp(x / 2 * (qq - 1)) - np.exp(x / 2 * (qq + 1)), 0, dtype=np.float32)
        r1 = lambda t: np.exp((qq - 1)/2 * xmin + (qq - 1)** 2 / 4 * t, dtype=np.float32)
        r2 = lambda t: 0.0
    boundary = 0.0
    for i in range(tstep-1, -1, -1):
        d[0], d[-1] = r1(tau * i)*theta+r1(tau*(i+1))*(1-theta), r2(tau * i)*theta+r2(tau*(i+1))*(1-theta)
        w = np.linalg.solve(A, np.matmul(B, w) + d)
        if opt_type == 'us_put':
            boundary = np.exp(-r * T * (tstep - i)) * (1 - np.exp(x)) / np.exp(-(qq - 1) / 2 * x - tau * (tstep - i) * ((qq - 1)** 2 / 4 + q), dtype=np.float32)
            w = np.maximum(w, boundary, dtype=np.float32)
        elif opt_type == 'us_call':
            boundary = np.exp(-r * T * (tstep - i)) * (np.exp(x)-1) / np.exp(-(qq - 1) / 2 * x - tau * (tstep - i) * ((qq - 1)** 2 / 4 + q), dtype=np.float32)
            w = np.maximum(w, boundary, dtype=np.float32)
    return w[xstep] * K * np.exp(-(qq - 1) / 2 * np.log(s0 / K) - tau*tstep * ((qq - 1)** 2 / 4 + q), dtype=np.float32)
