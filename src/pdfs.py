import numpy as np
from scipy.special import i0
from scipy.stats import norm
from scipy.special import gamma, beta

class PDFs:
    """Probability Density Functions (PDFs) class."""
    @staticmethod
    def alpha(x, a, loc=0, scale=1):
        """Alpha distribution."""
        z = (x - loc) / scale
        return (a / scale) * np.exp(-1 / z) / (z ** (a + 1)) * (z > 0)

    @staticmethod
    def anglit(x, loc=0, scale=1):
        """Anglit distribution."""
        z = (x - loc) / scale
        return (np.sin(2 * z) + (z >= -np.pi/4) * (z <= np.pi/4)) / scale

    @staticmethod
    def arcsine(x, loc=0, scale=1):
        """Arcsine distribution."""
        z = (x - loc) / scale
        return (1 / (np.pi * np.sqrt(z * (1 - z)))) * ((z > 0) & (z < 1)) / scale

    @staticmethod
    def beta(x, a, b, loc=0, scale=1):
        """Beta distribution."""
        z = (x - loc) / scale
        return (z**(a - 1) * (1 - z)**(b - 1) / beta(a, b)) * (z > 0) * (z < 1) / scale

    @staticmethod
    def betaprime(x, a, b, loc=0, scale=1):
        """Beta prime distribution."""
        z = (x - loc) / scale
        return (z**(a - 1) / (1 + z)**(a + b)) / (beta(a, b) * scale) * (z > 0)

    @staticmethod
    def bradford(x, c, loc=0, scale=1):
        """Bradford distribution."""
        z = (x - loc) / scale
        return c / (scale * (1 + c * z)) * (z > 0) * (z < 1)

    @staticmethod
    def burr(x, c, d, loc=0, scale=1):
        """Burr distribution."""
        z = (x - loc) / scale
        return (c * d * z**(c - 1) / (1 + z**c)**(d + 1)) / scale * (z > 0)

    @staticmethod
    def cauchy(x, loc=0, scale=1):
        """Cauchy distribution."""
        return (1 / (np.pi * scale)) * (scale**2 / ((x - loc)**2 + scale**2))

    @staticmethod
    def chi(x, df, loc=0, scale=1):
        """Chi distribution."""
        z = (x - loc) / scale
        return (1 / (2**(df/2 - 1) * gamma(df/2))) * z**(df - 1) * np.exp(-z**2 / 2) / scale

    @staticmethod
    def chi2(x, df, loc=0, scale=1):
        """Chi-squared distribution."""
        z = (x - loc) / scale
        return (1 / (2**(df/2) * gamma(df/2))) * z**(df/2 - 1) * np.exp(-z / 2) / scale

    @staticmethod
    def cosine(x, loc=0, scale=1):
        """Cosine distribution."""
        z = (x - loc) / scale
        return (np.cos(np.pi * z / 2) * (abs(z) <= 1)) / (2 * scale)

    @staticmethod
    def dgamma(x, a, loc=0, scale=1):
        """Double Gamma distribution."""
        z = (x - loc) / scale
        return (abs(z)**(a-1) * np.exp(-abs(z))) / (2 * gamma(a) * scale)

    @staticmethod
    def dweibull(x, c, loc=0, scale=1):
        """Double Weibull distribution."""
        z = (x - loc) / scale
        return (c / (2 * scale)) * abs(z)**(c - 1) * np.exp(-abs(z)**c)

    @staticmethod
    def erlang(x, k, loc=0, scale=1):
        """Erlang distribution."""
        z = (x - loc) / scale
        return (z**(k-1) * np.exp(-z)) / (gamma(k) * scale) * (z > 0)

    @staticmethod
    def expon(x, loc=0, scale=1):
        """Exponential distribution."""
        z = (x - loc) / scale
        return (1 / scale) * np.exp(-z) * (z >= 0)

    @staticmethod
    def exponnorm(x, K, loc=0, scale=1):
        """Exponentially modified Normal distribution."""
        z = (x - loc) / scale
        return (1 / (2 * K)) * np.exp(1 / (2 * K**2) - z / K) * norm.cdf(z / K - 1 / K)

    @staticmethod
    def exponweib(x, a, c, loc=0, scale=1):
        """Exponentiated Weibull distribution."""
        z = (x - loc) / scale
        return (a * c * (z ** (c - 1)) * np.exp(-z**c)) / scale * (1 - np.exp(-z**c))**(a - 1)

    @staticmethod
    def exponpow(x, b, loc=0, scale=1):
        """Exponential power distribution."""
        z = (x - loc) / scale
        return (b / scale) * np.exp(-z**b) * (z > 0)

    @staticmethod
    def f(x, dfn, dfd, loc=0, scale=1):
        """F-distribution."""
        z = (x - loc) / scale
        return ((dfn / dfd) ** (dfn / 2) * z**(dfn / 2 - 1) * (1 + dfn * z / dfd) ** (- (dfn + dfd) / 2)) / (beta(dfn / 2, dfd / 2) * scale) * (z > 0)

    @staticmethod
    def fatiguelife(x, c, loc=0, scale=1):
        """Fatigue-life (Birnbaum-Saunders) distribution."""
        z = (x - loc) / scale
        return (z + 1) / (2 * c * np.sqrt(2 * np.pi * z**3)) * np.exp(- (z - 1)**2 / (2 * c**2))

    @staticmethod
    def fisk(x, c, loc=0, scale=1):
        """Fisk (Log-logistic) distribution."""
        z = (x - loc) / scale
        return (c / scale) * (z**(c - 1) / (1 + z**c)**2) * (z > 0)

    @staticmethod
    def foldcauchy(x, c, loc=0, scale=1):
        """Folded Cauchy distribution."""
        z = (x - loc) / scale
        return (1 / (np.pi * scale * (1 + z**2))) * (z > 0)

    @staticmethod
    def foldnorm(x, c, loc=0, scale=1):
        """Folded Normal distribution."""
        z = (x - loc) / scale
        return (np.sqrt(2 / np.pi) * np.exp(-z**2 / 2)) / scale * (z > 0)

    @staticmethod
    def genlogistic(x, c, loc=0, scale=1):
        """Generalized Logistic distribution."""
        z = (x - loc) / scale
        return (np.exp(-z) / (scale * (1 + np.exp(-z))**(c + 1)))

    @staticmethod
    def genpareto(x, c, loc=0, scale=1):
        """Generalized Pareto distribution."""
        z = (x - loc) / scale
        return (1 / scale) * (1 + c * z) ** (-1 - 1 / c) * (1 + c * z > 0)

    @staticmethod
    def gennorm(x, beta, loc=0, scale=1):
        """Generalized Normal distribution."""
        z = (x - loc) / scale
        return (beta / (2 * scale * gamma(1 / beta))) * np.exp(-abs(z)**beta)

    @staticmethod
    def genexpon(x, a, b, c, loc=0, scale=1):
        """Generalized Exponential distribution."""
        z = (x - loc) / scale
        return (a * b * np.exp(-b * z) * (1 - np.exp(-b * z))**(a - 1) * np.exp(-c * z)) / scale * (z > 0)

    @staticmethod
    def genextreme(x, c, loc=0, scale=1):
        """Generalized Extreme Value distribution."""
        z = (x - loc) / scale
        return (np.exp(-z) * np.exp(-np.exp(-z))) / scale

    @staticmethod
    def gamma(x, a, loc=0, scale=1):
        """Gamma distribution."""
        z = (x - loc) / scale
        return (z**(a - 1) * np.exp(-z)) / (gamma(a) * scale) * (z > 0)

    @staticmethod
    def gengamma(x, a, c, loc=0, scale=1):
        """Generalized Gamma distribution."""
        z = (x - loc) / scale
        return (c / (scale * gamma(a))) * z**(a * c - 1) * np.exp(-z**c) * (z > 0)
    
    @staticmethod
    def genhalflogistic(x, c, loc=0, scale=1):
        """Generalized Half-Logistic distribution."""
        z = (x - loc) / scale
        return (c * np.exp(-z) / (scale * (1 + np.exp(-z))**(c + 1)))

    @staticmethod
    def gibrat(x, loc=0, scale=1):
        """Gibrat distribution."""
        z = (x - loc) / scale
        return (1 / (z * scale)) * np.exp(- (np.log(z))**2 / 2)

    @staticmethod
    def gompertz(x, c, loc=0, scale=1):
        """Gompertz distribution."""
        z = (x - loc) / scale
        return c * np.exp(z) * np.exp(-c * (np.exp(z) - 1)) / scale

    @staticmethod
    def gumbel_r(x, loc=0, scale=1):
        """Gumbel Right distribution."""
        z = (x - loc) / scale
        return (1 / scale) * np.exp(-(z + np.exp(-z)))

    @staticmethod
    def gumbel_l(x, loc=0, scale=1):
        """Gumbel Left distribution."""
        z = (x - loc) / scale
        return (1 / scale) * np.exp(z - np.exp(z))

    @staticmethod
    def halfcauchy(x, loc=0, scale=1):
        """Half-Cauchy distribution."""
        z = (x - loc) / scale
        return (2 / (np.pi * scale * (1 + z**2))) * (z > 0)

    @staticmethod
    def halflogistic(x, loc=0, scale=1):
        """Half-Logistic distribution."""
        z = (x - loc) / scale
        return (np.exp(-z) / (scale * (1 + np.exp(-z))**2)) * (z > 0)

    @staticmethod
    def halfnorm(x, loc=0, scale=1):
        """Half-Normal distribution."""
        z = (x - loc) / scale
        return (np.sqrt(2 / np.pi) * np.exp(-z**2 / 2) / scale) * (z > 0)

    @staticmethod
    def halfgennorm(x, beta, loc=0, scale=1):
        """Half-Generalized Normal distribution."""
        z = (x - loc) / scale
        return (beta / (scale * gamma(1 / beta))) * np.exp(-abs(z)**beta) * (z > 0)

    @staticmethod
    def hypsecant(x, loc=0, scale=1):
        """Hyperbolic Secant distribution."""
        z = (x - loc) / scale
        return (1 / (scale * np.pi)) * (1 / np.cosh(z))

    @staticmethod
    def invgamma(x, a, loc=0, scale=1):
        """Inverse Gamma distribution."""
        z = (x - loc) / scale
        return (z**(-a-1) * np.exp(-1 / z)) / (gamma(a) * scale) * (z > 0)

    @staticmethod
    def invgauss(x, mu, loc=0, scale=1):
        """Inverse Gaussian distribution."""
        z = (x - loc) / scale
        return (1 / np.sqrt(2 * np.pi * z**3)) * np.exp(-((z - mu)**2) / (2 * mu**2 * z))

    @staticmethod
    def invweibull(x, c, loc=0, scale=1):
        """Inverse Weibull distribution."""
        z = (x - loc) / scale
        return (c / scale) * (z**(-c - 1)) * np.exp(-z**-c) * (z > 0)

    @staticmethod
    def johnsonsb(x, a, b, loc=0, scale=1):
        """Johnson SB distribution."""
        z = (x - loc) / scale
        return (b / (scale * z * (1 - z))) * np.exp(-0.5 * ((a + b * np.log(z / (1 - z)))**2)) * (z > 0) * (z < 1)

    @staticmethod
    def johnsonsu(x, a, b, loc=0, scale=1):
        """Johnson SU distribution."""
        z = (x - loc) / scale
        return (b / (scale * np.sqrt(z**2 + 1))) * np.exp(-0.5 * ((a + b * np.log(z + np.sqrt(z**2 + 1)))**2))

    @staticmethod
    def laplace(x, loc=0, scale=1):
        """Laplace distribution."""
        z = (x - loc) / scale
        return (1 / (2 * scale)) * np.exp(-abs(z))

    @staticmethod
    def levy(x, loc=0, scale=1):
        """Levy distribution."""
        z = (x - loc) / scale
        return (1 / (np.sqrt(2 * np.pi * z**3))) * np.exp(-1 / (2 * z))

    @staticmethod
    def logistic(x, loc=0, scale=1):
        """Logistic distribution."""
        z = (x - loc) / scale
        return np.exp(-z) / (scale * (1 + np.exp(-z))**2)

    @staticmethod
    def loggamma(x, c, loc=0, scale=1):
        """Log-Gamma distribution."""
        z = (x - loc) / scale
        return (np.exp(-z - np.exp(-z)) / scale) * z**(c - 1)

    @staticmethod
    def loglaplace(x, c, loc=0, scale=1):
        """Log-Laplace distribution."""
        z = (x - loc) / scale
        return (c / (scale * z)) * (z**(c - 1) if z > 1 else z**(-c - 1))

    @staticmethod
    def lognorm(x, s, loc=0, scale=1):
        """Log-Normal distribution."""
        z = (x - loc) / scale
        return (1 / (z * s * np.sqrt(2 * np.pi))) * np.exp(- (np.log(z))**2 / (2 * s**2))

    @staticmethod
    def lomax(x, c, loc=0, scale=1):
        """Lomax distribution."""
        z = (x - loc) / scale
        return (c / scale) * (1 + z)**(-c - 1) * (z > 0)

    @staticmethod
    def maxwell(x, loc=0, scale=1):
        """Maxwell distribution."""
        z = (x - loc) / scale
        return (np.sqrt(2 / np.pi) * z**2 * np.exp(-z**2 / 2) / scale**3) * (z > 0)

    @staticmethod
    def mielke(x, k, s, loc=0, scale=1):
        """Mielke distribution."""
        z = (x - loc) / scale
        return (k * z**(k - 1)) / (1 + z**s)**(k + s) * (z > 0)

    @staticmethod
    def nakagami(x, nu, loc=0, scale=1):
        """Nakagami distribution."""
        z = (x - loc) / scale
        return (2 * nu**nu / gamma(nu)) * z**(2 * nu - 1) * np.exp(-nu * z**2) / scale**(2 * nu)

    @staticmethod
    def norm(x, loc=0, scale=1):
        """Normal (Gaussian) distribution."""
        return (1 / (np.sqrt(2 * np.pi) * scale)) * np.exp(-((x - loc) ** 2) / (2 * scale ** 2))
    
    @staticmethod
    def pareto(x, b, loc=0, scale=1):
        """Pareto distribution."""
        z = (x - loc) / scale
        return (b / scale) * (z**(-b - 1)) * (z > 1)

    @staticmethod
    def pearson3(x, skew, loc=0, scale=1):
        """Pearson Type III distribution."""
        z = (x - loc) / scale
        return norm.pdf(z) * (1 + skew * (z - skew / 2))

    @staticmethod
    def powerlaw(x, a, loc=0, scale=1, const=1):
        """Power-law distribution."""
        z = (x - loc) / scale
        return const * (z**(a)) * (z > 0)
    
    @staticmethod
    def powerlaw_simple(x, a, scale=1,):
        """Power-law distribution."""
        return scale * (x**a)

    @staticmethod
    def powerlognorm(x, c, s, loc=0, scale=1):
        """Power Log-Normal distribution."""
        z = (x - loc) / scale
        return (c / (z * s * np.sqrt(2 * np.pi))) * np.exp(- (np.log(z))**2 / (2 * s**2)) * (z > 0)

    @staticmethod
    def powernorm(x, c, loc=0, scale=1):
        """Power Normal distribution."""
        z = (x - loc) / scale
        return (c * np.exp(-z**2 / 2)) / (scale * np.sqrt(2 * np.pi)) * (z > 0)

    @staticmethod
    def rayleigh(x, loc=0, scale=1):
        """Rayleigh distribution."""
        z = (x - loc) / scale
        return (z / scale**2) * np.exp(-z**2 / (2 * scale**2)) * (z > 0)

    @staticmethod
    def rice(x, nu, loc=0, scale=1):
        """Rice distribution."""
        z = (x - loc) / scale
        return (z / scale**2) * np.exp(- (z**2 + nu**2) / (2 * scale**2)) * i0(z * nu / scale**2) * (z > 0)

    @staticmethod
    def semicircular(x, loc=0, scale=1):
        """Semicircular distribution."""
        z = (x - loc) / scale
        return (2 / (np.pi * scale)) * np.sqrt(1 - z**2) * (abs(z) < 1)

    @staticmethod
    def t(x, df, loc=0, scale=1):
        """Student's T distribution."""
        z = (x - loc) / scale
        return (gamma((df + 1) / 2) / (np.sqrt(df * np.pi) * gamma(df / 2))) * (1 + z**2 / df)**(-(df + 1) / 2)

    @staticmethod
    def rdist(x, c, loc=0, scale=1):
        """R-distribution."""
        z = (x - loc) / scale
        return ((c + 1) / (2 * scale)) * (1 - abs(z)**c) * (abs(z) < 1)

    @staticmethod
    def reciprocal(x, a, b, loc=0, scale=1):
        """Reciprocal distribution."""
        z = (x - loc) / scale
        return (1 / (z * np.log(b / a))) * ((z >= a) & (z <= b))

    @staticmethod
    def recipinvgauss(x, mu, loc=0, scale=1):
        """Reciprocal Inverse Gaussian distribution."""
        z = (x - loc) / scale
        return (np.exp(- (mu / z - 1)**2 / (2 * mu**2 * z))) / (np.sqrt(2 * np.pi) * z**1.5)

    @staticmethod
    def triang(x, c, loc=0, scale=1):
        """Triangular distribution."""
        z = (x - loc) / scale
        return (2 / scale) * (z * (z < c) / c + (1 - z) * (z >= c) / (1 - c)) * ((z >= 0) & (z <= 1))

    @staticmethod
    def truncexpon(x, b, loc=0, scale=1):
        """Truncated Exponential distribution."""
        z = (x - loc) / scale
        return (np.exp(-z) / (scale * (1 - np.exp(-b)))) * ((z >= 0) & (z <= b))

    @staticmethod
    def truncnorm(x, a, b, loc=0, scale=1):
        """Truncated Normal distribution."""
        z = (x - loc) / scale
        return (norm.pdf(z) / (scale * (norm.cdf(b) - norm.cdf(a)))) * ((z >= a) & (z <= b))

    @staticmethod
    def tukeylambda(x, lam, loc=0, scale=1):
        """Tukey-Lambda distribution."""
        z = (x - loc) / scale
        return ((1 - np.abs(z)**lam) ** (1 / lam - 1)) / scale * (np.abs(z) < 1)
    
    @staticmethod
    def uniform(x, loc=0, scale=1):
        """Uniform distribution."""
        z = (x - loc) / scale
        return (1 / scale) * ((z >= 0) & (z <= 1))

    @staticmethod
    def vonmises(x, kappa, loc=0, scale=1):
        """Von Mises distribution."""
        z = (x - loc) / scale
        return (np.exp(kappa * np.cos(z)) / (2 * np.pi * i0(kappa)))

    @staticmethod
    def wald(x, loc=0, scale=1):
        """Wald (Inverse Gaussian) distribution."""
        z = (x - loc) / scale
        return (1 / np.sqrt(2 * np.pi * z**3)) * np.exp(- (z - scale)**2 / (2 * z * scale**2))

    @staticmethod
    def weibull_min(x, c, loc=0, scale=1):
        """Weibull Minimum distribution."""
        z = (x - loc) / scale
        return (c / scale) * z**(c - 1) * np.exp(-z**c) * (z > 0)

    @staticmethod
    def weibull_max(x, c, loc=0, scale=1):
        """Weibull Maximum distribution."""
        z = (loc - x) / scale
        return (c / scale) * z**(c - 1) * np.exp(-z**c) * (z > 0)

    @staticmethod
    def wrapcauchy(x, c, loc=0, scale=1):
        """Wrapped Cauchy distribution."""
        z = (x - loc) / scale
        return (1 - c**2) / (2 * np.pi * (1 + c**2 - 2 * c * np.cos(z)))
    
