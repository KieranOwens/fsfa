# Kieran Owens 2025
# fsfa: feature-based slow feature analysis

##############################################################################
# 1. Dependencies
##############################################################################
import numpy as np
from scipy.signal import stft
import pandas as pd
from pycatch22 import catch22_all
import catch22_C
from sksfa import SFA
from features import get_default_features

##############################################################################
# 2. Features
##############################################################################

# catch22 library full feature names
features_c22 = [
        'DN_HistogramMode_5',
        'DN_HistogramMode_10',
        'CO_f1ecac',
        'CO_FirstMin_ac',
        'CO_HistogramAMI_even_2_5',
        'CO_trev_1_num',
        'MD_hrv_classic_pnn40',
        'SB_BinaryStats_mean_longstretch1',
        'SB_TransitionMatrix_3ac_sumdiagcov',
        'PD_PeriodicityWang_th0_01',
        'CO_Embed2_Dist_tau_d_expfit_meandiff',
        'IN_AutoMutualInfoStats_40_gaussian_fmmi',
        'FC_LocalSimple_mean1_tauresrat',
        'DN_OutlierInclude_p_001_mdrmd',
        'DN_OutlierInclude_n_001_mdrmd',
        'SP_Summaries_welch_rect_area_5_1',
        'SB_BinaryStats_diff_longstretch0',
        'SB_MotifThree_quantile_hh',
        'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
        'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
        'SP_Summaries_welch_rect_centroid',
        'FC_LocalSimple_mean3_stderr'
        ]

# catchaMouse16 full feature names
features_cm16 = [
        'SY_DriftingMean50_min',
        'DN_RemovePoints_absclose_05_ac2rat',
        'AC_nl_036',
        'AC_nl_112',
        'ST_LocalExtrema_n100_diffmaxabsmin',
        'IN_AutoMutualInfoStats_diff_20_gaussian_ami8',
        'CO_HistogramAMI_even_2_3',
        'CO_TranslateShape_circle_35_pts_statav4_m',
        'CO_AddNoise_1_even_10_ami_at_10',
        'PH_Walker_momentum_5_w_momentumzcross',
        'SC_FluctAnal_2_dfa_50_2_logi_r2_se2',
        'PH_Walker_biasprop_05_01_sw_meanabsdiff',
        'CO_HistogramAMI_even_10_3',
        'AC_nl_035',
        'FC_LoopLocalSimple_mean_stderr_chn',
        'CO_TranslateShape_circle_35_pts_std'
	    ]


FEATURE_DICT = {
    'catch24': {'names': ['mode_5', 'mode_10', 'acf_timescale', 'acf_first_min', 'ami2',
                    'trev', 'high_fluctuation', 'stretch_high', 'transition_matrix', 
                    'periodicity', 'embedding_dist', 'ami_timescale', 'whiten_timescale',
                    'outlier_timing_pos', 'outlier_timing_neg', 'centroid_freq',
                    'stretch_decreasing', 'entropy_pairs', 'rs_range', 'dfa', 
                    'low_freq_power', 'forecast_error', 'mean', 'std'],
                'features': [getattr(catch22_C, f) for f in features_c22] + [np.mean, np.var]},

    'catch22': {'names': ['mode_5', 'mode_10', 'acf_timescale', 'acf_first_min', 'ami2',
                    'trev', 'high_fluctuation', 'stretch_high', 'transition_matrix', 
                    'periodicity', 'embedding_dist', 'ami_timescale', 'whiten_timescale',
                    'outlier_timing_pos', 'outlier_timing_neg', 'centroid_freq',
                    'stretch_decreasing', 'entropy_pairs', 'rs_range', 'dfa', 
                    'low_freq_power', 'forecast_error'],
                'features': [getattr(catch22_C, f) for f in features_c22]},

    'meanvar': {'names': ['mean', 'variance'], 
                'features': [np.mean, np.var]},

    'mean': {'names': ['mean'], 
             'features': [np.mean]}
}

try:

    import catchaMouse16_C

    f_catchaMouse16 = [getattr(catchaMouse16_C, f) for f in features_cm16]

    FEATURE_DICT['catchaMouse16'] = {'names': ['stationarity_min', 'outlier_corr', 'nonlin_autocorr_036',
                                     'nonlin_autocorr_112', 'outlier_asymmetry', 'increment_ami8',
                                     'ami3_2bin', 'stationarity_floating_circle', 'noise_titration',
                                     'walker_crossings', 'dfa_longscale_fit', 'walker_diff', 'ami3_10bin',
                                     'nonlin_autocorr_035', 'prediction_scale', 'floating_circle'],
                                     'features': f_catchaMouse16
                                    }
except:
  
    print("Module 'catchaMouse16_C' is unavailable. The catchaMouse16 feature set cannot be used.")


##############################################################################
# 3. Feature embedding
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
# 4. Feature-based slow feature analysis
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