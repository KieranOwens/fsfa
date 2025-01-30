# Kieran Owens 2025
# fsfa: feature-based slow feature analysis

##############################################################################
# 1. Dependencies
##############################################################################
import numpy as np
from scipy.signal import stft
import pandas as pd
from pycatch22 import catch22_all
from sksfa import SFA
from features import get_default_features

FEATURE_DICT = get_default_features()

##############################################################################
# 2. Feature embedding
##############################################################################

def feature_embedding(X,
                      window,
                      stride,
                      featureset='catch24',
                      featurenames=None,
                      variablenames=None,
                      nb_quantiles=20,
                      fs=1.0,
                      stft_stride=1,
                      verbose=False):
    """
    Given a (possibly multivariate) time series input, extract a (possibly multivariate)
    time series of statistical time-series features.

    Parameters
    ----------
    X : numpy array
        A (possibly multivariate) time series of dimension (nb_samples, nb_variables).

    window : int
        The length of the sliding window over which statistics will be computed.

    stride : int
        The stride length used for sliding window computation of statistics, i.e.,
        the number array indices between each adjacent window.

    featureset: str or list of functions
        A time-series feature set can be specified using a string, with options
        'mean', 'meanvar', 'quantiles', 'stft', 'catch22', 'catch24', and
        'catchaMouse16' .The catchaMouse options require separate installation 
        of the catchaMouse16 package. Alternatively, a list of time-series 
        statistical functions can be provided.
        Default: 'catch24'.

    featurenames: list of strings
        An optional list of time-series feature names which must be of the same
        length as the list of functions provided in the argument featureset.
        Default: None.

    variablenames: list of strings
        An optional list of variable names.
        Default:
        None.

    verbose: boolean
        Determines whether a message is provided to the user.
        Default: False.

    nb_quantiles: int
        Determines how many quantiles to compute per window if using quantile 
        features.
        Default: 20.

    fs: float
        Specifies the sampling frequency of the underlying time series. This 
        is only used when computing spectrogram features via the short-time 
        Fourier transform (STFT).
        Default: 1.0.

    stft_stride: int
        If set to N it returns every Nth STFT frequency feature. 
        Default: 1.
        
    Returns
    -------
    A time series of statistical time-series features,
    """

    # number of samples, number of variables
    T, D = X.shape

    # time points at which to evaluate window features
    time_points = np.arange(0, T - window + 1, stride)

    # window indices for feature calculations
    window_indices = np.array([np.arange(t, t + window) for t in time_points])

    ###############
    # features
    ###############

    if featureset in FEATURE_DICT:

        featurenames = FEATURE_DICT[featureset]['names']

        featureset = FEATURE_DICT[featureset]['features']

    elif featurenames == None:

        featurenames = [f'f{i}' for i in range(len(featureset))]

    ###############
    # variable names
    ###############
    # default is ['x0', 'x1', 'x2', ... ]
    if variablenames == None:

        variablenames = [f'x{i}' for i in range(D)]

    ###############
    # compute features
    ###############

    # compute quantiles
    if featureset == 'quantiles':

        quantiles = np.linspace(0, 1, nb_quantiles)

        F = np.zeros((len(window_indices), D * nb_quantiles))

        for i, indices in enumerate(window_indices):

            for j in range(D):

                F[i,j*nb_quantiles:(j+1)*nb_quantiles] = np.quantile(X[indices,j], q=quantiles)

        featurenames = [f'{var}_q{i}' for var in variablenames 
                                      for i in range(nb_quantiles)]
        
        F = pd.DataFrame(data=F, 
                         columns=featurenames)

    # or compute stft
    elif featureset == 'stft':

        noverlap = window - stride

        F = []

        for i in range(D):

            # Compute the Short-Time Fourier Transform (STFT)
            f, _, zxx = stft(X[:,i], fs, nperseg=window, noverlap=noverlap, 
                             boundary=None, padded=False)

            # Compute the spectrogram power spectrum
            sxx = np.abs(zxx) ** 2

            # Convert the power spectrum to decibels (dB)
            sxx_db = 20 * np.log(sxx / np.amax(sxx))

            F += [sxx_db[::stft_stride,:].T]

        featurenames = [f'{var}_fx_{fx}' for var in variablenames 
                                      for fx in f[::stft_stride]]

        F = pd.DataFrame(data=np.concatenate(F, axis=1), 
                         columns=featurenames)
    
    # or compute catch22/24 
    elif featureset in ['catch22', 'catch24']:

        catch24 = (featureset == 'catch24')

        F = []

        for i in range(D):
            Fi = np.array([catch22_all(X[idx, i], catch24=catch24)['values'] for idx in window_indices])
            F.append(Fi)

        featurenames = [f'{var}_{name}' for var in variablenames 
                                      for name in featurenames]
        
        F = pd.DataFrame(data=np.concatenate(F, axis=1), 
                         columns=featurenames)


    # or compute from a list of features
    else:

        # dictionary to store time series of statistics
        results_dict = {}

        for i, x in enumerate(variablenames):

            for j, f in enumerate(featureset):

                var_feat = f'{x}_{featurenames[j]}'

                if verbose:
                    print(f"Calculating {var_feat}...")

                results_dict[var_feat] = [f(list(X[indices, i])) for indices in window_indices]

        F = pd.DataFrame.from_dict(results_dict)

    ###############
    # return data frame
    ###############

    return F

##############################################################################
# 3. Feature-based slow feature analysis
##############################################################################

def fSFA(X,
         window,
         stride,
         featureset='catch24',
         featurenames=None,
         variablenames=None,
         nb_quantiles=20,
         fs=1.0,
         stft_stride=1,
         n_components=1,
         kwargs = dict(),
         verbose=False):
    """
    Feature-based Slow Feature Analysis (f-SFA) is an approach to Parameter
    Inference from a Non-stationary Unknown Process (PINUP), involving 
    (1) sliding-window computation of a multivariate time series of statistical
    time series features and (2) dimension reduction via SFA. 
    
    The output of f-SFA represents a subspaces that capture the directions of 
    slowest statistical variation for the input time series. The resulting 
    f-SFA time series can be interpreted as tracking the underlying parameter(s) 
    driving non-stationary variation in statistical properties of the input 
    time series.

    Parameters
    ----------
    X : numpy array
        A (possibly multivariate) time series of dimension (nb_samples, nb_variables).

    window : int
        The length of the sliding window over which statistics will be computed.

    stride : int
        The stride length used for sliding window computation of statistics, i.e.,
        the number array indices between each adjacent window.

    featureset: str or list of functions
        A time-series feature set can be specified using a string, with options
        'mean', 'meanvar', 'quantiles', 'stft', 'catch22', 'catch24', and
        'catchaMouse16' .The catchaMouse options require separate installation 
        of the catchaMouse16 package. Alternatively, a list of time-series 
        statistical functions can be provided.
        Default: 'catch24'.

    featurenames: list of strings
        An optional list of time-series feature names which must be of the same
        length as the list of functions provided in the argument featureset.
        Default: None.

    variablenames: list of strings
        An optional list of variable names.
        Default:
        None.

    verbose: boolean
        Determines whether a message is provided to the user.
        Default: False.

    nb_quantiles: int
        Determines how many quantiles to compute per window if using quantile 
        features.
        Default: 20.

    fs: float
        Specifies the sampling frequency of the underlying time series. This 
        is only used when computing spectrogram features via the short-time 
        Fourier transform (STFT).
        Default: 1.0.

    stft_stride: int
        If set to N it returns every Nth STFT frequency feature. 
        Default: 1.

    n_components: int
        The number of components to return in the SFA dimension reduction step.
        default: 1

    kwargs: dict
        A dictionary of keyword arguments to provide to the sksfa SFA function.
        Including 'svd_solver', 'tol', 'robustness_cutoff', and 'fill_mode'.
        See the sksfa SFA documentation for more details:
        https://sklearn-sfa.readthedocs.io/en/latest/index.html

    verbose: boolean
        Determines whether a message is provided to the user.
        Default: False.

    Returns
    -------
    Component time series of SFA applied to a time series of statistical 
    time-series features.
    """
    
    F = feature_embedding(X, window, stride, 
                          featureset=featureset,
                          featurenames=featurenames,
                          variablenames=variablenames,
                          nb_quantiles=nb_quantiles,
                          fs=fs,
                          stft_stride=stft_stride,
                          verbose=verbose)

    return SFA(n_components=n_components, **kwargs).fit_transform(F)