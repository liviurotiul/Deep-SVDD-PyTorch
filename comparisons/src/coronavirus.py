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
from sklearn.feature_extraction import image
import torch
import pickle
from skimage.feature import hog
import seaborn as sns

def mod(x):
    if x == -1:
        return 0
    return x


image_test_path = "/home/liviu/Documents/Dev/2nHack/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/image_test_dict.pkl"
image_train_path = "/home/liviu/Documents/Dev/2nHack/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/image_train_dict.pkl"
label_path = "/home/liviu/Documents/Dev/2nHack/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/label_dict.pkl"

image_dict = {}

with open(image_train_path, "rb") as image_train_dict:
    image_dict = pickle.load(image_train_dict)

with open(image_test_path, "rb") as image_test_dict:
    image_dict.update(pickle.load(image_test_dict))


label_dict = open(label_path, "rb")
label_dict = pickle.load(label_dict)

table = []
features = []
labels = []

for key in image_dict:
    if key == 'X_ray_image_name':
        continue
    try:
        temp = label_dict[key]
    except:
        continue

    image = image_dict[key]

    label = None
    if temp[2] == 'COVID-19':
        label = 1
        # print(temp)
    else:
        label = 0

    table.append([label, image])
image_dict = None
label_dict = None




features = [row[1] for row in table]
features = [np.array(x.convert('L')) for x in features]


ppc = 24
hog_images = []
hog_features = []
for image in features:
    fd = hog(image, orientations=12, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2')
    hog_features.append(fd)



features = hog_features

labels = [row[0] for row in table]
table = None

for i in range(0,1500):
    if labels[i] == 1:
        labels[i], labels[i+4000] = labels[i+4000], labels[i]
        features[i], features[i+4000] = features[i+4000], features[i]
        print("swaped")


train_data = features[0:4000]
train_labels = labels[0:4000]

test_data = features[4000:5911]
test_labels = labels[4000:5911]



# train_data = features[0:1000]
# train_labels = labels[0:1000]

# test_data = features[4000:4111]
# test_labels = labels[4000:4111]



print("training OCSVM")
OCSVM = OneClassSVM(gamma='auto', nu=0.02).fit(train_data)
print("training isolation forest")
isolation_forest = IsolationForest(random_state=0).fit(train_data)
print("training local outliar")
local_outliar = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(train_data)


# fig = plt.figure(figsize=(15, 10))
# ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
#                    xticklabels=[], ylim=(-1.2, 1.2))
# ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
#                    ylim=(-1.2, 1.2))


models = [OCSVM, isolation_forest, local_outliar]
names = ["ocsvm", "isolationForest", "localOutliar"]
for model, name in zip(models, names):
    scores = None
    print("===================================================================================")
    print(name)
    predicted = model.predict(test_data)

    scores = 1-model.score_samples(test_data)
    predicted = [mod(x) for x in predicted]
    # print(predicted[0:10])
    # print(scores[0:10])

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



    fpr, tpr, threshold = roc_curve(test_labels, scores)
    roc_auc = auc(fpr, tpr)

    # plt.figure(figsize=(15, 10))

    
    # sns.lineplot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc).set_title('Receiver Operating Characteristic')
    # # sns.legend(loc = 'lower right')
    # sns.lineplot([0, 1], [0, 1])
    # sns.xlim([0, 1])
    # sns.ylim([0, 1])
    # sns.ylabel('True Positive Rate')
    # sns.xlabel('False Positive Rate')
    # plt.show()

    nonzero_indeces = np.nonzero(test_labels)[0]
    zero_indeces = np.where(test_labels == 0)[0]

    outliars = scores[nonzero_indeces]
    normal_samples = scores[zero_indeces]
    # plt.figure(figsize=(15, 10))
    # sns.displot(scores, kind="kde")
    # sns.displot(outliars, kind="kde")
    # plt.hist(scores, color='green')
    # plt.figure(figsize=(20, 20))
    # scores
    # # plt.hist(outliars, color='blue')

    # plt.show()
    np.save("/home/liviu/Documents/Dev/Deep-SVDD-PyTorch/results/coronavirus/"+name+"_fpr", fpr)
    np.save("/home/liviu/Documents/Dev/Deep-SVDD-PyTorch/results/coronavirus/"+name+"_tpr", tpr)
    np.save("/home/liviu/Documents/Dev/Deep-SVDD-PyTorch/results/coronavirus/"+name+"_scores", scores)
    np.save("/home/liviu/Documents/Dev/Deep-SVDD-PyTorch/results/coronavirus/"+name+"_outliars", outliars)
