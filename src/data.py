import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import pandas as pd
# sns.set_style("white")
# plt.figure(figsize=(15, 10))

path = "/home/liviu/Documents/Dev/Deep-SVDD-PyTorch/results"
for folder in os.listdir(path):

    print(folder)

    DeepSVDD_fpr = np.load(path+'/'+folder+'/'+'DeepSVDD_fpr'+'.npy')
    ocsvm_fpr = np.load(path+'/'+folder+'/'+'ocsvm_fpr'+'.npy')
    isolation_forest_fpr = np.load(path+'/'+folder+'/'+'isolation_forest_fpr'+'.npy')
    local_outliar_fpr = np.load(path+'/'+folder+'/'+'local_outliar_fpr'+'.npy')

    DeepSVDD_tpr = np.load(path+'/'+folder+'/'+'DeepSVDD_tpr'+'.npy')
    ocsvm_tpr = np.load(path+'/'+folder+'/'+'ocsvm_tpr'+'.npy')
    isolation_forest_tpr = np.load(path+'/'+folder+'/'+'isolation_forest_tpr'+'.npy')
    local_outliar_tpr = np.load(path+'/'+folder+'/'+'local_outliar_tpr'+'.npy')

    DeepSVDD_scores = np.load(path+'/'+folder+'/'+'DeepSVDD_scores'+'.npy')
    ocsvm_scores = np.load(path+'/'+folder+'/'+'ocsvm_scores'+'.npy')
    isolation_forest_scores = np.load(path+'/'+folder+'/'+'isolation_forest_scores'+'.npy')
    local_outliar_scores = np.load(path+'/'+folder+'/'+'local_outliar_scores'+'.npy')

    DeepSVDD_outliars = np.load(path+'/'+folder+'/'+'DeepSVDD_outliars'+'.npy')
    ocsvm_outliars = np.load(path+'/'+folder+'/'+'ocsvm_outliars'+'.npy')
    isolation_forest_outliars = np.load(path+'/'+folder+'/'+'isolation_forest_outliars'+'.npy')
    local_outliar_outliars = np.load(path+'/'+folder+'/'+'local_outliar_outliars'+'.npy')


    plt.figure(figsize=(15, 10))
    plt.title(folder+"_scor_AUC")
    roc_auc_DeepSVDD = auc(DeepSVDD_fpr, DeepSVDD_tpr)
    roc_auc_OCSVM = auc(ocsvm_fpr, ocsvm_tpr)
    roc_auc_isolation = auc(isolation_forest_fpr, isolation_forest_tpr)
    roc_auc_local = auc(local_outliar_fpr, local_outliar_tpr)

    plt.plot(DeepSVDD_fpr, DeepSVDD_tpr, color='blue', label="Deep SVDD %0.2f" % roc_auc_DeepSVDD+'%')
    plt.plot(ocsvm_fpr, ocsvm_tpr, color='green', label="OCSVM %0.2f" % roc_auc_OCSVM+'%')
    plt.plot(isolation_forest_fpr, isolation_forest_tpr, color='black', label="Isolation Forest %0.2f" % roc_auc_isolation+'%')
    plt.plot(local_outliar_fpr, local_outliar_tpr, color='yellow', label="Local Outliar %0.2f" % roc_auc_local+'%')

    plt.legend(loc="upper left")

    plt.show()



    plt.figure(figsize=(15, 10))
    plt.title(folder+"_histograma_scoruri")
    temp = [DeepSVDD_scores, DeepSVDD_outliars]
    for i in range(0,len(temp)):
        temp[i] = temp[i]-np.min(temp[i])
        temp[i] = temp[i]/10

    sns.displot(temp, kind="kde", common_norm=False, multiple="fill")

    plt.legend(loc="center right", labels=['DeepSVDD_scores','DeepSVDD_outliars'])

    plt.show()
    plt.close()

    plt.figure(figsize=(15, 10))
    plt.title(folder+"_histograma_scoruri")
    temp = [ocsvm_scores, ocsvm_outliars]
    for i in range(0,len(temp)):
        temp[i] = temp[i]-np.min(temp[i])
        temp[i] = temp[i]/10

    sns.displot(temp, kind="kde", common_norm=False, multiple="fill")

    plt.legend(loc="center right", labels=['ocsvm_scores','ocsvm_outliars'])

    plt.show()


    plt.figure(figsize=(15, 10))
    plt.title(folder+"_histograma_scoruri")
    temp = [isolation_forest_scores, isolation_forest_outliars]
    for i in range(0,len(temp)):
        temp[i] = temp[i]-np.min(temp[i])
        temp[i] = temp[i]/10

    sns.displot(temp, kind="kde", common_norm=False, multiple="fill")

    plt.legend(loc="center right", labels=['isolation_forest_scores','isolation_forest_outliars'])

    plt.show()



    plt.figure(figsize=(15, 10))
    plt.title(folder+"_histograma_scoruri")
    temp = [local_outliar_scores, local_outliar_outliars]
    for i in range(0,len(temp)):
        temp[i] = temp[i]-np.min(temp[i])
        temp[i] = temp[i]/10

    sns.displot(temp, kind="kde", common_norm=False, multiple="fill")

    plt.legend(loc="center right", labels=['local_outliar_scores','local_outliar_outliars'])

    plt.show()
