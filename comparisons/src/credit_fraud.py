import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
path_2_csv = "/home/liviu/Documents/Dev/Deep-SVDD-PyTorch/Datasets/CreditFraud/creditcard.csv"

def mod(x):
    if x == -1:
        return 0
    return x


# Path to the csv file
path_2_csv = "/home/liviu/Documents/Dev/Deep-SVDD-PyTorch/Datasets/malware_detection/tek_data.csv"

# Open and read the csv
with open(path_2_csv, mode='r') as infile:
    reader = csv.reader(infile)
    table = []
    for row in reader:
        table.append(list(row))

table = table[1:-1]        
table = [[float(x) for x in row] for row in table]

table = np.asarray(table)
for i, _ in enumerate(table[0]):
    table[:,i] = table[:,i]/np.max(table[:,i])
np.random.shuffle(table)
features = table[:,1:56]
labels = table[:,-1]
# features[:,0] = features[:,0]/np.max(features[:,0])
# print(features[])
# import pdb; pdb.set_trace()
# self.train_data = torch.from_numpy(features[0:200000])
# self.test_data = torch.from_numpy(features[200001:284807])
# self.train_labels = torch.from_numpy(labels[0:200000])
# self.test_labels = torch.from_numpy(labels[200001:284807])

train_data = torch.from_numpy(features[0:100000])
test_data = torch.from_numpy(features[100000:138046])
train_labels = torch.from_numpy(labels[0:100000])
test_labels = torch.from_numpy(labels[100000:138046])

# knn = KNeighborsClassifier(n_neighbors=10, algorithm='ball_tree').fit(train_data, train_labels)
# kmeans = KMeans(n_clusters=2, random_state=0).fit(train_data)
OCSVM = OneClassSVM(gamma='auto', nu=0.02).fit(train_data)
isolation_forest = IsolationForest(random_state=0).fit(train_data)
local_outliar = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(train_data)

models = [local_outliar]
names = ["local_outliar"]
for model, name in zip(models, names):

    print("===================================================================================")
    print(name)
    predicted = model.predict(test_data)
    scores = 1-model.score_samples(test_data)
    predicted = [mod(x) for x in predicted]
    print(predicted[0:10])
    print(scores[0:10])

    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0


    for (item1, item2) in zip(predicted, test_labels):
        if item1 == item2 == 1:
            true_positive += 1
        if item1 == item2 == 0:
            true_negative += 1
        if item1 == 1 and item2 == 0:
            false_positive += 1
        if item1 == 0 and item2 == 1:
            false_negative += 1

    print("true_positive: ", true_positive)
    print("true_negative: ", true_negative)
    print("false_positive: ", false_positive)
    print("false_negative: ", false_negative)


    # scores = model.predict_proba(test_data)
    # print(scores[:,1])
    # print(predicted)

    fpr, tpr, threshold = roc_curve(test_labels, scores)
    roc_auc = auc(fpr, tpr)

    # fpr, tpr, threshold = roc_curve(test_labels, scores[:,1])
    # roc_auc = auc(fpr, tpr)
    # print(roc_auc)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
