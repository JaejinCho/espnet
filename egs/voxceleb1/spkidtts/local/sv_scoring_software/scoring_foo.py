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

    C_Miss = np.array([1,1])
    C_FalseAlarm = np.array([1,1])
    P_Target = np.array([0.5, 0.01])
    beta = (C_FalseAlarm/C_Miss)*((1-P_Target)/P_Target)
    ln_beta = np.log(beta)

    number_of_nontarget, number_of_target = np.bincount(labels)
    # Scoring global without partionning
    min_dcf_1_unequalized = scr.compute_c_norm(fnr_unequalized, fpr_unequalized, P_Target[0])
    min_dcf_2_unequalized = scr.compute_c_norm(fnr_unequalized, fpr_unequalized, P_Target[1])

    print("EER: {0:.2f} minDCF1e-3: {1:.4f} minDCF1e-2: {2:.4f}".format(eer * 100, min_dcf_1_unequalized, min_dcf_2_unequalized))
