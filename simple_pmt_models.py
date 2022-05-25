import numpy as np
import scipy.stats as ss

def _shorten_and_flip_pmf(q, qsummax=0.9999999999):
    return np.flipud(q[0:min(np.count_nonzero(np.cumsum(q)<qsummax)+1,len(q))])

def _cascade_pmf(q, Fp, p0=0):
    return np.polyval(q, Fp), np.polyval(q, p0)

def _rebin(p, binsize):
    return np.diff(np.interp(np.arange(0,len(p),binsize),
        np.maximum(np.arange(len(p)+1)-0.5, 0), np.append(0,np.cumsum(p))))


def cascaded_pmt_spectrum(lambda_2K, lambda_1_hi, nstage, npmf, lambda_0=None,
                            p_1_lo=None, sigma_1_lo=0, e0=0, sigma_e=None,
                            suppress_zero=False, electrons_per_dc=None,
                            interim_rebin_nstage=None, interim_rebin_factor=0):
    """ Calculate the spectrum of a PMT using a cascaded, multistage, Poisson
    model including photo-electron statistics and Gaussian noise. The first
    stage of the PMT amplifier can include a low-charge, half-Gaussian
    population. See Fegan (2022)

    Parameters
    ----------
    lambda_2K : Mean of the Poisson distribution for amplification stages 2..K

    lambda_1_hi : Mean of the Poisson distrbution for the (high-charge component
        of the) first amplification stage. Value must be in range [0,inf]

    nstage : number of amplification stages

    npmf : number of points to compute in the spectrum. Should be chosen large
        enough to ensure that the probability distribution is well contained
        in spectrum, otherwise cyclic leackage will occur

    lambda_0 : Mean of Poisson distribution for photo-cathode, or None if
        photo-cathode is not to be included in model

    p_1_lo : probability of PE being amplified by low-charge component of 1st
        stage. Value must be in range [0,1]

    sigma_1_lo : width of the zero-centered Gaussian describing the low-charge
        component of the 1st stage

    suppress_zero : suppress events that result in zero output electrons being
        produced after amplification

    electrons_per_dc : enables rebinning of the output spectrum with a given
        number of electrons per digital count. The value of electrons_per_dc
        must be greater than the parameter "interim_rebin_factor" for it to
        have any effect.

    interim_rebin_nstage : specify stage after which to apply interim rebinning
        if non-zero. The value must be less than or equal to K-1. Interim
        rebinning can yield a significant reduction in computation by allowing
        "npmf" to be reduced (it can normally be reduced by a factor of
        1/interim_rebin_nstage, if care is takem)

    interim_rebin_factor : size of bins to apply in interim rebinning. Must be
        greater than 1.0 to have an effect.

    """
    x = np.arange(npmf+1)
    k = 2*np.pi*np.arange(npmf//2+1)/npmf

    # Initialize Fourier transform of PMF to that of unit impulse (Kronicker delta)
    Fp, p0 = np.cos(k) - 1j*np.sin(k), 0

    # Compute coefficients of Poisson PMF for stages 2..K and reduce length
    q = _shorten_and_flip_pmf(ss.poisson.pmf(x[:-1], lambda_2K))
    if(interim_rebin_nstage is not None and interim_rebin_nstage>0 and interim_rebin_factor>1.0):
        # Apply interim rebinning by calculating response of "interim_rebin_nstage"
        # final stages, and rebinning to give PMF in units of "interim_rebin_factor"
        assert interim_rebin_nstage <= nstage-1
        Fp_temp = Fp
        for i in range(interim_rebin_nstage):
            Fp_temp, _ = _cascade_pmf(q, Fp_temp)
        p = _rebin(np.fft.irfft(Fp_temp, n=npmf),interim_rebin_factor)
        Fp, p0 = _cascade_pmf(_shorten_and_flip_pmf(p, qsummax=0.99999999), Fp, p0)
    else:
        interim_rebin_nstage = 0
        interim_rebin_factor = 1.0
    # Calculate response (spectrum) of stages 2..K
    for i in range(nstage-1-interim_rebin_nstage):
        Fp, p0 = _cascade_pmf(q, Fp, p0)

    # Compute coefficients of PMF for stage 1, including high-charge (Poisson)
    # and low-charge (half-Gaussian) components
    q = ss.poisson.pmf(x[:-1], lambda_1_hi)
    if(p_1_lo is not None and p_1_lo>0 and sigma_1_lo>0):
        q_lo = np.diff(ss.norm.cdf(np.maximum(x-0.5,0),loc=0,scale=sigma_1_lo))/0.5
        q = min(p_1_lo,1.0)*q_lo + max(1-p_1_lo,0)*q

    # Calculate full response (spectrum) of amplifier, stages 1..K
    Fp, p0 = _cascade_pmf(_shorten_and_flip_pmf(q), Fp, p0)

    # Apply suppression of zero-charge events if desired
    if(suppress_zero):
        Fp = (Fp - p0)/(1 - p0)

    # Calculate spectrum of PMT including statistics on PEs in photo-cathode
    if(lambda_0 is not None and lambda_0 >= 0):
        q = _shorten_and_flip_pmf(ss.poisson.pmf(x[:-1], lambda_0))
        Fp, _ = _cascade_pmf(q, Fp)

    # Add Gaussian noise by multiplying by the Fourier transform of a Gaussian,
    # (which is itself a Gaussian) translated by the correct phase factors
    if(sigma_e is not None and sigma_e > 0):
        if(electrons_per_dc is not None and electrons_per_dc>interim_rebin_factor):
            sigma_e *= electrons_per_dc/interim_rebin_factor
            e0 *= electrons_per_dc/interim_rebin_factor
        Fp *= np.exp(-0.5*k**2*(sigma_e**2)) * (np.cos(k*e0) - 1j*np.sin(k*e0))

    # Transform PMF back from Fourier space
    p = np.fft.irfft(Fp, n=npmf)

    if(electrons_per_dc is not None and electrons_per_dc>interim_rebin_factor):
        p = _rebin(p, electrons_per_dc/interim_rebin_factor)

    return p
