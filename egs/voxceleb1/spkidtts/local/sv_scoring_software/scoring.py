# -*- coding: utf-8 -*-

"""
This script computes the official performance metrics for the NIST 2016 SRE.
The metrics include EER and DCFs (min/act).

"""

__author__  = "Timothee Kheyrkhah"
__email__   = "timothee.kheyrkhah@nist.gov"
__version__ = "4.2"

import os # os.system("pause") for windows command line
import sys
import time
#import configparser as cp
#import argparse
import numpy as np

#import matplotlib.pyplot as plt
import scoring_tools as st
import sre_score as scr

if __name__ == '__main__':
    scores = []
    labels = []
    trials_fi = open(sys.argv[1], 'r').readlines()
    scores_fi = open(sys.argv[2], 'r').readlines()
    spkrutt2target = {}
    for line in trials_fi:
      spkr, utt, target = line.strip().split()
      spkrutt2target[spkr+utt]=target
    for line in scores_fi:
      spkr, utt, score = line.strip().split()
      if spkr+utt in spkrutt2target:
        target = spkrutt2target[spkr+utt]
        scores.append(float(score))
        if target == "nontarget":
          labels.append(0)
        else:
          labels.append(1)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    fnr_unequalized, fpr_unequalized = scr.compute_pmiss_pfa_rbst(scores, labels)
    eer = scr.compute_eer(fnr_unequalized, fpr_unequalized)

    # Boolean for disabling the command line
    debug_mode_ide = False
    show_minDCF_1_2 = True
    plot_det = False

    C_Miss = np.array([1,1,1,1])
    C_FalseAlarm = np.array([1,1,1,1])
    P_Target = np.array([0.05,0.01,0.005,0.001])
    beta = (C_FalseAlarm/C_Miss)*((1-P_Target)/P_Target)
    ln_beta = np.log(beta)

    number_of_nontarget, number_of_target = np.bincount(labels)
    # Scoring global without partionning

    # min_dcf_2_unequalized = scr.compute_c_norm(fnr_unequalized, fpr_unequalized, P_Target[1])
    # min_dcf_3_unequalized = scr.compute_c_norm(fnr_unequalized, fpr_unequalized, P_Target[2])
    # min_dcf_4_unequalized = scr.compute_c_norm(fnr_unequalized, fpr_unequalized, P_Target[3])

    min_dcf=[]
    act_dcf=[]
    for i in range(len(P_Target)):
        min_dcf_i = scr.compute_c_norm(fnr_unequalized, fpr_unequalized, P_Target[i])        
        act_dcf_i,P_Miss_Target,P_FalseAlarm_NonTarget = st.actual_detection_cost(
            scores,labels,number_of_nontarget,number_of_target,1,[beta[i]],[ln_beta[i]])
        min_dcf.append(min_dcf_i)
        act_dcf.append(act_dcf_i)
     
    print("EER: {0:.2f} DCF5e-2: {1:.3f} / {2:.3f} DCF1e-2: {3:.3f} / {4:.3f} DCF5e-3: {5:.3f} / {6:.3f} DCF1e-3: {7:.3f} / {8:.3f}".format(
        eer * 100, min_dcf[0], act_dcf[0],
        min_dcf[1], act_dcf[1],
        min_dcf[2], act_dcf[2],
        min_dcf[3], act_dcf[3])) 
