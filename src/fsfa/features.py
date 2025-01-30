# Kieran Owens 2025
# fsfa: feature-based slow feature analysis

##############################################################################
# 1. Dependencies
##############################################################################
import numpy as np
import catch22_C

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

def get_default_features():

    return FEATURE_DICT