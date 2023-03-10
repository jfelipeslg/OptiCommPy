"""Functions for adaptive and static equalization."""
import logging as logg

import numpy as np
import scipy.constants as const
from numba import njit
from numpy.fft import fft, fftfreq, ifft
from tqdm.notebook import tqdm

from optic.dsp import pnorm, clippingComplex
from optic.models import linFiberCh
from optic.modulation import GrayMapping
from optic.metrics import signal_power


def edc(Ei, L, D, Fc, Fs):
    """
    Electronic chromatic dispersion compensation (EDC).

    Parameters
    ----------
    Ei : np.array
        Dispersed signal.
    L : real scalar
        Fiber length [km].
    D : real scalar
        Chromatic dispersion parameter [ps/nm/km].
    Fc : real scalar
        Carrier frequency [Hz].
    Fs : real scalar
        Sampling frequency [Hz].

    Returns
    -------
    np.array
        CD compensated signal.

    """
    return linFiberCh(Ei, L, 0, -D, Fc, Fs)


def mimoAdaptEqualizer(x, dx=[], paramEq=[]):
    """
    N-by-N MIMO adaptive equalizer.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    dx : TYPE, optional
        DESCRIPTION. The default is [].
    paramEq : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    yEq : TYPE
        DESCRIPTION.
    H : TYPE
        DESCRIPTION.
    errSq : TYPE
        DESCRIPTION.
    Hiter : TYPE
        DESCRIPTION.

    """
    # check input parameters
    numIter = getattr(paramEq, "numIter", 1)
    nTaps = getattr(paramEq, "nTaps", 15)
    mu = getattr(paramEq, "mu", [1e-3])
    lambdaRLS = getattr(paramEq, "lambdaRLS", 0.99)
    SpS = getattr(paramEq, "SpS", 2)
    H = getattr(paramEq, "H", [])
    L = getattr(paramEq, "L", [])
    Hiter = getattr(paramEq, "Hiter", [])
    storeCoeff = getattr(paramEq, "storeCoeff", False)
    alg = getattr(paramEq, "alg", ["nlms"])
    constType = getattr(paramEq, "constType", "qam")
    M = getattr(paramEq, "M", 4)
    prgsBar = getattr(paramEq, "prgsBar", True)

    # We want all the signal sequences to be disposed in columns:
    if not len(dx):
        dx = x.copy()
    try:
        if x.shape[1] > x.shape[0]:
            x = x.T
    except IndexError:
        x = x.reshape(len(x), 1)
    try:
        if dx.shape[1] > dx.shape[0]:
            dx = dx.T
    except IndexError:
        dx = dx.reshape(len(dx), 1)
    nModes = int(x.shape[1])  # number of sinal modes (order of the MIMO equalizer)

    Lpad = int(np.floor(nTaps / 2))
    zeroPad = np.zeros((Lpad, nModes), dtype="complex")
    x = np.concatenate(
        (zeroPad, x, zeroPad)
    )  # pad start and end of the signal with zeros

    # Defining training parameters:
    constSymb = GrayMapping(M, constType)  # constellation
    constSymb = pnorm(constSymb)  # normalized constellation symbols

    totalNumSymb = int(np.fix((len(x) - nTaps) / SpS + 1))

    if not L:  # if L is not defined
        L = [
            totalNumSymb
        ]  # Length of the output (1 sample/symbol) of the training section
    if not H:  # if H is not defined
        H = np.zeros((nModes ** 2, nTaps), dtype="complex")

        for initH in range(nModes):  # initialize filters' taps
            H[
                initH + initH * nModes, int(np.floor(H.shape[1] / 2))
            ] = 1  # Central spike initialization
    # Equalizer training:
    if type(alg) == list:

        yEq = np.zeros((totalNumSymb, x.shape[1]), dtype="complex")
        errSq = np.zeros((totalNumSymb, x.shape[1])).T

        nStart = 0
        for indstage, runAlg in enumerate(alg):
            logg.info(f"{runAlg} - training stage #%d", indstage)

            nEnd = nStart + L[indstage]

            if indstage == 0:
                for indIter in tqdm(range(numIter), disable=not (prgsBar)):
                    logg.info(
                        f"{runAlg} pre-convergence training iteration #%d", indIter
                    )
                    yEq[nStart:nEnd, :], H, errSq[:, nStart:nEnd], Hiter = coreAdaptEq(
                        x[nStart * SpS : nEnd * SpS, :],
                        dx[nStart:nEnd, :],
                        SpS,
                        H,
                        L[indstage],
                        mu[indstage],
                        lambdaRLS,
                        nTaps,
                        storeCoeff,
                        runAlg,
                        constSymb,
                    )
                    logg.info(
                        f"{runAlg} MSE = %.6f.", np.nanmean(errSq[:, nStart:nEnd])
                    )
            else:
                yEq[nStart:nEnd, :], H, errSq[:, nStart:nEnd], Hiter = coreAdaptEq(
                    x[nStart * SpS : nEnd * SpS, :],
                    dx[nStart:nEnd, :],
                    SpS,
                    H,
                    L[indstage],
                    mu[indstage],
                    lambdaRLS,
                    nTaps,
                    storeCoeff,
                    runAlg,
                    constSymb,
                )
                logg.info(f"{runAlg} MSE = %.6f.", np.nanmean(errSq[:, nStart:nEnd]))
            nStart = nEnd
    else:
        for indIter in tqdm(range(numIter), disable=not (prgsBar)):
            logg.info(f"{alg}training iteration #%d", indIter)
            yEq, H, errSq, Hiter = coreAdaptEq(
                x, dx, SpS, H, L, mu, nTaps, storeCoeff, alg, constSymb
            )
            logg.info(f"{alg}MSE = %.6f.", np.nanmean(errSq))
    return yEq, H, errSq, Hiter


@njit
def coreAdaptEq(x, dx, SpS, H, L, mu, lambdaRLS, nTaps, storeCoeff, alg, constSymb):
    """
    Adaptive equalizer core processing function
    
    """

    # allocate variables
    nModes = int(x.shape[1])
    indTaps = np.arange(0, nTaps)
    indMode = np.arange(0, nModes)

    errSq = np.empty((nModes, L))
    yEq = x[:L].copy()
    yEq[:] = np.nan
    outEq = np.array([[0 + 1j * 0]]).repeat(nModes).reshape(nModes, 1)

    if storeCoeff:
        Hiter = (
            np.array([[0 + 1j * 0]])
            .repeat((nModes ** 2) * nTaps * L)
            .reshape(nModes ** 2, nTaps, L)
        )
    else:
        Hiter = (
            np.array([[0 + 1j * 0]])
            .repeat((nModes ** 2) * nTaps)
            .reshape(nModes ** 2, nTaps, 1)
        )
    if alg == "rls":
        Sd = np.eye(nTaps, dtype=np.complex128)
        a = Sd.copy()
        for _ in range(nTaps - 1):
            Sd = np.concatenate((Sd, a))
    # Radii cma, rde
    Rcma = (
        np.mean(np.abs(constSymb) ** 4) / np.mean(np.abs(constSymb) ** 2)
    ) * np.ones((1, nModes)) + 1j * 0
    Rrde = np.unique(np.abs(constSymb))

    for ind in range(L):
        outEq[:] = 0

        indIn = indTaps + ind * SpS  # simplify indexing and improve speed

        # pass signal sequence through the equalizer:
        for N in range(nModes):
            inEq = x[indIn, N].reshape(
                len(indIn), 1
            )  # slice input coming from the Nth mode
            outEq += (
                H[indMode + N * nModes, :] @ inEq
            )  # add contribution from the Nth mode to the equalizer's output
        yEq[ind, :] = outEq.T

        # update equalizer taps acording to the specified
        # algorithm and save squared error:
        if alg == "nlms":
            H, errSq[:, ind] = nlmsUp(x[indIn, :], dx[ind, :], outEq, mu, H, nModes)
        elif alg == "cma":
            H, errSq[:, ind] = cmaUp(x[indIn, :], Rcma, outEq, mu, H, nModes)
        elif alg == "dd-lms":
            H, errSq[:, ind] = ddlmsUp(x[indIn, :], constSymb, outEq, mu, H, nModes)
        elif alg == "rde":
            H, errSq[:, ind] = rdeUp(x[indIn, :], Rrde, outEq, mu, H, nModes)
        elif alg == "da-rde":
            H, errSq[:, ind] = dardeUp(x[indIn, :], dx[ind, :], outEq, mu, H, nModes)
        elif alg == "rls":
            H, Sd, errSq[:, ind] = rlsUp(
                x[indIn, :], dx[ind, :], outEq, lambdaRLS, H, Sd, nModes
            )
        elif alg == "dd-rls":
            H, Sd, errSq[:, ind] = ddrlsUp(
                x[indIn, :], constSymb, outEq, lambdaRLS, H, Sd, nModes
            )
        elif alg == "static":
            errSq[:, ind] = errSq[:, ind - 1]
        else:
            raise ValueError(
                "Equalization algorithm not specified (or incorrectly specified)."
            )
        if storeCoeff:
            Hiter[:, :, ind] = H
        else:
            Hiter[:, :, 1] = H
    return yEq, H, errSq, Hiter


@njit
def nlmsUp(x, dx, outEq, mu, H, nModes):
    """
    coefficient update with the NLMS algorithm    
    """
    indMode = np.arange(0, nModes)
    err = dx - outEq.T  # calculate output error for the NLMS algorithm

    errDiag = np.diag(err[0])  # define diagonal matrix from error array

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing and improve speed
        inAdapt = x[:, N].T / np.linalg.norm(x[:, N]) ** 2  # NLMS normalization
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(x), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * errDiag @ np.conj(inAdaptPar)
        )  # gradient descent update
    return H, np.abs(err) ** 2


@njit
def rlsUp(x, dx, outEq, λ, H, Sd, nModes):
    """
    coefficient update with the RLS algorithm    
    """
    nTaps = H.shape[1]
    indMode = np.arange(0, nModes)
    indTaps = np.arange(0, nTaps)

    err = dx - outEq.T  # calculate output error for the NLMS algorithm

    errDiag = np.diag(err[0])  # define diagonal matrix from error array

    # update equalizer taps
    for N in range(nModes):
        indUpdModes = indMode + N * nModes
        indUpdTaps = indTaps + N * nTaps

        Sd_ = Sd[indUpdTaps, :]

        inAdapt = np.conj(x[:, N]).reshape(-1, 1)  # input samples
        inAdaptPar = (
            (inAdapt.T).repeat(nModes).reshape(len(x), -1).T
        )  # expand input to parallelize tap adaptation

        Sd_ = (1 / λ) * (
            Sd_
            - (Sd_ @ (inAdapt @ (np.conj(inAdapt).T)) @ Sd_)
            / (λ + (np.conj(inAdapt).T) @ Sd_ @ inAdapt)
        )

        H[indUpdModes, :] += errDiag @ (Sd_ @ inAdaptPar.T).T

        Sd[indUpdTaps, :] = Sd_
    return H, Sd, np.abs(err) ** 2


@njit
def ddlmsUp(x, constSymb, outEq, mu, H, nModes):
    """
    coefficient update with the DD-LMS algorithm    
    """
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    decided = np.zeros(outEq.shape, dtype=np.complex128)

    for k in range(nModes):
        indSymb = np.argmin(np.abs(outEq[0, k] - constSymb))
        decided[0, k] = constSymb[indSymb]
    err = decided - outEq  # calculate output error for the DDLMS algorithm

    errDiag = np.diag(err[0])  # define diagonal matrix from error array

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = x[:, N].T
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(x), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * errDiag @ np.conj(inAdaptPar)
        )  # gradient descent update
    return H, np.abs(err) ** 2


@njit
def ddrlsUp(x, constSymb, outEq, λ, H, Sd, nModes):
    """
    coefficient update with the DD-RLS algorithm    
    """
    nTaps = H.shape[1]
    indMode = np.arange(0, nModes)
    indTaps = np.arange(0, nTaps)

    outEq = outEq.T
    decided = np.zeros(outEq.shape, dtype=np.complex128)

    for k in range(nModes):
        indSymb = np.argmin(np.abs(outEq[0, k] - constSymb))
        decided[0, k] = constSymb[indSymb]
    err = decided - outEq  # calculate output error for the DDLMS algorithm

    errDiag = np.diag(err[0])  # define diagonal matrix from error array

    # update equalizer taps
    for N in range(nModes):
        indUpdModes = indMode + N * nModes
        indUpdTaps = indTaps + N * nTaps

        Sd_ = Sd[indUpdTaps, :]

        inAdapt = np.conj(x[:, N]).reshape(-1, 1)  # input samples
        inAdaptPar = (
            (inAdapt.T).repeat(nModes).reshape(len(x), -1).T
        )  # expand input to parallelize tap adaptation

        Sd_ = (1 / λ) * (
            Sd_
            - (Sd_ @ (inAdapt @ (np.conj(inAdapt).T)) @ Sd_)
            / (λ + (np.conj(inAdapt).T) @ Sd_ @ inAdapt)
        )

        H[indUpdModes, :] += errDiag @ (Sd_ @ inAdaptPar.T).T

        Sd[indUpdTaps, :] = Sd_
    return H, Sd, np.abs(err) ** 2


@njit
def cmaUp(x, R, outEq, mu, H, nModes):
    """
    coefficient update with the CMA algorithm    
    """
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    err = R - np.abs(outEq) ** 2  # calculate output error for the CMA algorithm

    prodErrOut = np.diag(err[0]) @ np.diag(outEq[0])  # define diagonal matrix

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = x[:, N].T
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(x), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * prodErrOut @ np.conj(inAdaptPar)
        )  # gradient descent update
    return H, np.abs(err) ** 2


@njit
def rdeUp(x, R, outEq, mu, H, nModes):
    """
    coefficient update with the RDE algorithm    
    """
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    decidedR = np.zeros(outEq.shape, dtype=np.complex128)

    # find closest constellation radius
    for k in range(nModes):
        indR = np.argmin(np.abs(R - np.abs(outEq[0, k])))
        decidedR[0, k] = R[indR]
    err = (
        decidedR ** 2 - np.abs(outEq) ** 2
    )  # calculate output error for the RDE algorithm

    prodErrOut = np.diag(err[0]) @ np.diag(outEq[0])  # define diagonal matrix

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = x[:, N].T
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(x), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * prodErrOut @ np.conj(inAdaptPar)
        )  # gradient descent update
    return H, np.abs(err) ** 2


@njit
def dardeUp(x, dx, outEq, mu, H, nModes):
    """
    coefficient update with the data-aided RDE algorithm    
    """
    indMode = np.arange(0, nModes)
    outEq = outEq.T
    decidedR = np.zeros(outEq.shape, dtype=np.complex128)

    # find exact constellation radius
    for k in range(nModes):
        decidedR[0, k] = np.abs(dx[k])
    err = (
        decidedR ** 2 - np.abs(outEq) ** 2
    )  # calculate output error for the RDE algorithm

    prodErrOut = np.diag(err[0]) @ np.diag(outEq[0])  # define diagonal matrix

    # update equalizer taps
    for N in range(nModes):
        indUpdTaps = indMode + N * nModes  # simplify indexing
        inAdapt = x[:, N].T
        inAdaptPar = (
            inAdapt.repeat(nModes).reshape(len(x), -1).T
        )  # expand input to parallelize tap adaptation
        H[indUpdTaps, :] += (
            mu * prodErrOut @ np.conj(inAdaptPar)
        )  # gradient descent update
    return H, np.abs(err) ** 2


def dbp(Ei, Fs, Ltotal, Lspan, hz=0.5, alpha=0.2, gamma=1.3, D=16, Fc=193.1e12):
    """
    Digital backpropagation (symmetric, single-pol.)

    :param Ei: input signal
    :param Ltotal: total fiber length [km]
    :param Lspan: span length [km]
    :param hz: step-size for the split-step Fourier method [km][default: 0.5 km]
    :param alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
    :param D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
    :param gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
    :param Fc: carrier frequency [Hz][default: 193.1e12 Hz]
    :param Fs: sampling frequency [Hz]

    :return Ech: backpropagated signal
    """
    # c = 299792458   # speed of light (vacuum)
    c_kms = const.c / 1e3
    λ = c_kms / Fc
    α = -alpha / (10 * np.log10(np.exp(1)))
    β2 = (D * λ ** 2) / (2 * np.pi * c_kms)
    γ = -gamma

    Nfft = len(Ei)

    ω = 2 * np.pi * Fs * fftfreq(Nfft)

    Nspans = int(np.floor(Ltotal / Lspan))
    Nsteps = int(np.floor(Lspan / hz))

    Ech = Ei.reshape(len(Ei),)
    Ech = fft(Ech)  # single-polarization field

    linOperator = np.exp(-(α / 2) * (hz / 2) + 1j * (β2 / 2) * (ω ** 2) * (hz / 2))

    for _ in tqdm(range(Nspans)):
        Ech = Ech * np.exp((α / 2) * Nsteps * hz)

        for _ in range(Nsteps):
            # First linear step (frequency domain)
            Ech = Ech * linOperator

            # Nonlinear step (time domain)
            Ech = ifft(Ech)
            Ech = Ech * np.exp(1j * γ * (Ech * np.conj(Ech)) * hz)

            # Second linear step (frequency domain)
            Ech = fft(Ech)
            Ech = Ech * linOperator
    Ech = ifft(Ech)

    return Ech.reshape(len(Ech),)


@njit
def DFR(R1, R2, sigLO):
    """
    Direct Field Reconstruction (DFR)
    
    Algorithm
    ---------
    1. Define function delta
    2. Calculation of in-phase and quadrature components (roots)

    Parameters
    ----------
    R1 : np.array
        Input in-phase component photocurrent.
    R2 : np.array
        Input in-quadracture component photocurrent.
    sigLO : np.array
        Input local oscillator (LO).
      
    Returns
    -------
    sigOut : np.array
        Output signal without interference beat signal-signal (SSBI) using the DFR method.

    """
    A = sigLO
    
    delta = 4*R1*R2 - (R1 + R2 - 2*A**2)**2
    
    sigI = - A/2 + 1/(4*A) * (R1 - R2) + 1/(4*A) * np.sqrt(delta)
    sigQ = - A/2 - 1/(4*A) * (R1 - R2) + 1/(4*A) * np.sqrt(delta)
      
    sigOut = sigI + 1j*sigQ
    
    return sigOut


@njit
def IC(R1, R2, sigWDM, sigLO, N=20, clipping=True):
    """
    Iterative SSBI Cancellation (IC)

    Algorithm
    ---------
    1. Change of variables
    2. Definition of analysis equations
    3. Initial estimation of signal and error
    4. Definition of the iterative loop for the estimations.

    Parameters
    ----------
    R1 : np.array
        Input in-phase component photocurrent.
    R2 : np.array
        Input in-quadracture component photocurrent.
    sigWDM : np.array
        Input received optical signal. 
    sigLO : np.array
        Input local oscillator (LO).
    N : scalar
        Number of loop repetitions. The default is 20.
    clipping : bool
        Clipping status. The default is True.
      
    Returns
    -------
    sigOut : np.array
        Output signal without interference beat signal-signal (SSBI) using the IC method.

    """
    A = sigLO                
    P = signal_power(sigWDM) 
    
    U1 = (R1 - A**2) / (4*A**2) 
    U2 = (R2 - A**2) / (4*A**2) 
    
    overline_I = sigWDM.real / (2*A) 
    overline_Q = sigWDM.imag / (2*A) 
        
    overline_I0 = U1 - P / (4*A**2) 
    overline_Q0 = U2 - P / (4*A**2) 
    e0 = overline_I0 - overline_I   

    overline_In = overline_I0 
    overline_Qn = overline_Q0 
    error = e0                
    
    if clipping == True:
        LOSPR = 10*np.log10( signal_power(sigLO) / P )                     
        optimumClip_dB  = LOSPR - 1                                         
        optimumClip_lin = 10**(optimumClip_dB/10)                           
        clippingValue   = optimumClip_lin * np.sqrt(signal_power(sigWDM)) 

    for nSteps in range(1, N):
        reductionSSBI  = (overline_In**2 + overline_Qn**2)      
        
        if clipping == True:
            reductionSSBI  = clippingComplex( reductionSSBI, clippingValue)
        
        overline_Inext = U1 - reductionSSBI                     
        overline_Qnext = U2 - reductionSSBI                     
        
        errorNext = - error * (overline_I + overline_Q + overline_In + overline_Qn) 
                
        overline_In = overline_Inext 
        overline_Qn = overline_Qnext 
        error = errorNext            
    
    overline_sigOut = (overline_In + 1j*overline_Qn) 
    sigOut = (2*A) * overline_sigOut                 
    
    return sigOut


@njit
def gradientDescent(R1, R2, sigWDM, sigLO, mu=0.05, N=150, clipping=True):
    """
    Gradient Descent (GD)

    Algorithm
    ---------
    1. Change of variables
    2. Equations of the GD algorithm
    3. Definition of the objective function
    4. Iterative update for minimizing the objective function.

    Parameters
    ----------
    R1 : np.array
        Input in-phase component photocurrent.
    R2 : np.array
        Input in-quadracture component photocurrent.
    sigWDM : np.array
        Input received optical signal. 
    sigLO : np.array
        Input local oscillator (LO).
    mu : scalar
        Convergence step. The default is 0.05.
    N : scalar
        Number of loop repetitions. The default is 150.
    clipping : bool
        Clipping status. The default is True.
      
    Returns
    -------
    sigOut : np.array
        Output signal without interference beat signal-signal (SSBI) using the GD method.

    """
    A = sigLO                
    P = signal_power(sigWDM) 
    
    U1 = (R1 - A**2) / (4*A**2)
    U2 = (R2 - A**2) / (4*A**2) 
    
    overline_I = sigWDM.real / (2*A) 
    overline_Q = sigWDM.imag / (2*A) 
    
    overline_I0 = U1 - P / (4*A**2) 
    overline_Q0 = U2 - P / (4*A**2)
    
    overline_In = overline_I0 
    overline_Qn = overline_Q0 
    
    if clipping == True:
        LOSPR = 10*np.log10( signal_power(sigLO) / signal_power(sigWDM) )       
        optimumClip_dB  = LOSPR + 4                                            
        optimumClip_lin = 10**(optimumClip_dB/10)                               
        clippingValueI   = optimumClip_lin * np.sqrt(signal_power(sigWDM.real)) 
        clippingValueQ   = optimumClip_lin * np.sqrt(signal_power(sigWDM.imag)) 
    
    for nSteps in range(1, N):
               
        X_InQn = overline_In**2 + overline_Qn**2 + overline_In - U1 
        Y_InQn = overline_In**2 + overline_Qn**2 + overline_Qn - U2 
        
        G = X_InQn ** 2 + Y_InQn ** 2
            
        gradientI = (X_InQn * (2*overline_In + 1) + 2 * Y_InQn * overline_In)
        gradientQ = (X_InQn * overline_Qn + 2 * Y_InQn * (2*overline_Qn + 1)) 
        
        if clipping == True:
            gradientI  = clippingComplex( gradientI, clippingValueI)
            gradientQ  = clippingComplex( gradientQ, clippingValueQ)
            
        overline_Inext = overline_In - mu * gradientI
        overline_Qnext = overline_Qn - mu * gradientQ
        
        error = 2 * np.sqrt( (overline_In - overline_I)**2 + (overline_Qn - overline_Q)**2 )
        
        overline_In = overline_Inext
        overline_Qn = overline_Qnext
        
    overline_sigOut = (overline_In + 1j*overline_Qn) 
    sigOut = (2*A) * overline_sigOut                 
    
    return sigOut


def mitigationSSBI(R1, R2, sigWDM, sigLO, paramEqSSBI=[]):
    """
    SSBI mitigation block
    
    Parameters
    ----------
    R1 : np.array
        Input in-phase component photocurrent.
    R2 : np.array
        Input in-quadracture component photocurrent.
    sigWDM : np.array
        Input received optical signal. 
    sigLO : np.array
        Input local oscillator (LO).
    paramEqSSBI : parameter object, optional
        Parameters of the SSBI mitigation algorithm.
      
    Returns
    -------
    sigRx : np.array
        Output signal without/with interference beat signal-signal (SSBI).

    """
    alg = getattr(paramEqSSBI, "alg", "dfr").lower()
    
    # Algorithm check
    assert alg in ["none", "dfr", "ic", "gd"], "Invalid SSBI mitigation algorithm."
    
    # Direct Field Reconstruction (DFR)
    if alg == "dfr":
        sigDFR = DFR(R1, R2, sigLO)
        sigRx  = sigDFR - np.mean(sigDFR)

    # Iterative SSBI Cancellation (IC)
    elif alg == "ic":
        # Check input parameters
        Nsteps     = getattr(paramEqSSBI, "Nsteps", 20)
        enableClip = getattr(paramEqSSBI, "enableClip", True)
        
        sigRx = IC(R1, R2, sigWDM, sigLO, N=Nsteps, clipping=enableClip)
    
    # Gradient Descent (GD)
    elif alg == "gd":
        # Check input parameters
        Nsteps     = getattr(paramEqSSBI, "Nsteps", 150)
        mu_        = getattr(paramEqSSBI, "mu", 0.05)
        enableClip = getattr(paramEqSSBI, "enableClip", True)
        
        sigRx = gradientDescent(R1, R2, sigWDM, sigLO, mu=mu_, N=Nsteps, clipping=enableClip)

    # Not applied SSBI mitigating algorithm
    else:
        sigPD = R1 + 1j*R2
        sigRx = sigPD - np.mean(sigPD)
        
    return sigRx
