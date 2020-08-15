
import cv2
import csv
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split,KFold
TRAIN_FOLDER = "/Users/priyarajpurohit/Desktop/Desktop – Priya’s MacBook Pro/bonus-sml-2020/SML_Train/Train_"
TRAIN_LABEL = "/Users/priyarajpurohit/Desktop/Desktop – Priya’s MacBook Pro/bonus-sml-2020/SML_Train.csv"
TEST_FOLDER = "/Users/priyarajpurohit/Desktop/Desktop – Priya’s MacBook Pro/bonus-sml-2020/SML_Test/Test_"
TEST_LABEL = "/Users/priyarajpurohit/Desktop/Desktop – Priya’s MacBook Pro/bonus-sml-2020/2015073_Priya_submission1.csv"
test_label="/Users/priyarajpurohit/Desktop/Desktop – Priya’s MacBook Pro/bonus-sml-20200/f2015073_model8.csv"
model_label="2015073_model8.pkl"
X_train_raw = []
y_train_raw = []
X_test_raw = []
y_test_raw = []
for i in range(16000):
    f = TRAIN_FOLDER + str(i) + ".jpg"
    #print(cv2.imread(f))
    im = cv2.imread(f).flatten()
    X_train_raw.append(im)
for i in range(1500):
    f = TEST_FOLDER + str(i) + ".jpg"
    im = cv2.imread(f).flatten()
    X_test_raw.append(im)

with open(TRAIN_LABEL, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    head = csvreader.__next__()
    for row in csvreader:
        y_train_raw.append(row[1])
csvfile.close()
# feature_scaler = StandardScaler()
# X_train_raw = feature_scaler.fit_transform(X_train_raw)
# X_test_raw = feature_scaler.transform(X_test_raw)
#
# pca = PCA(n_components=50)
# pca.fit(X_train_raw)
# X_train_raw = pca.transform(X_train_raw)
# X_test_raw = pca.transform(X_test_raw)
#
X_train1=[]
y_train1=[]
X_val=[]
y_val=[]

#classes=np.unique(y_train_raw)
# weights = compute_class_weight('balanced', classes, y_train_raw)
# d={}
# for i in range(len(weights)):
#     d[i]=weights[i]
# print(d)
X_train1, X_val, y_train1, y_val = train_test_split(X_train_raw, y_train_raw, test_size=0.2, random_state=1)
print("yay")

rf_class= RandomForestClassifier(n_estimators=200,max_depth=30, random_state=0,class_weight='balanced')
print(cross_val_score(rf_class, X_train1, y_train1, scoring='accuracy', cv = 10))
accuracy = cross_val_score(rf_class, X_train1, y_train1, scoring='accuracy', cv = 10).mean() * 100
print("Accuracy of Random Forests is: " , accuracy)
if(accuracy>20):
    print("YAY")
    rf_class.fit(X_train1, y_train1)
    s=rf_class.score(X_val,y_val)
    print(s)
    if s>0.222:
        rf_class.fit(X_train_raw,y_train_raw)
        predicted= rf_class.predict(X_test_raw)


        pickle.dump(rf_class, open(model_label, 'wb'))
        with open(test_label, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["id","category"])
            for x in range(len(predicted)):
                csvwriter.writerow(["Test_"+str(x)+".jpg",predicted[x]])
        csvfile.close()
        with open(TEST_LABEL, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["id","category"])
            for x in range(len(predicted)):
                csvwriter.writerow(["Test_"+str(x)+".jpg",predicted[x]])
        csvfile.close()
    else:
        print("nope")
