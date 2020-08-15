import cv2
import csv
import pickle
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage import filters
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split,KFold
TRAIN_FOLDER = "/Users/priyarajpurohit/Desktop/bonus-sml-2020/SML_Train/Train_"
TRAIN_LABEL = "/Users/priyarajpurohit/Desktop/bonus-sml-2020/SML_Train.csv"
TEST_FOLDER = "/Users/priyarajpurohit/Desktop/bonus-sml-2020/SML_Test/Test_"
TEST_LABEL = "/Users/priyarajpurohit/Desktop/bonus-sml-2020/2015073_Priya_submission1.csv"
test_label="/Users/priyarajpurohit/Desktop/bonus-sml-2020/rf_sobel.csv"
model_label="rf_sobel.pkl"
X_train_raw = []
SOBEL_x_train=[]
SOBEL_x_test=[]
y_train_raw = []
X_test_raw = []
y_test_raw = []
for i in range(16000):
    f = TRAIN_FOLDER + str(i) + ".jpg"
    #print(cv2.imread(f))
    im1 = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
    fd = filters.sobel(im1)
    SOBEL_x_train.append(fd.flatten())
    # im = cv2.imread(f).flatten()
    # X_train_raw.append(im)
for i in range(1500):
    f = TEST_FOLDER + str(i) + ".jpg"
    im1 = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
    fd = filters.sobel(im1)
    SOBEL_x_test.append(fd.flatten())
    # im = cv2.imread(f).flatten()
    # X_test_raw.append(im)

with open(TRAIN_LABEL, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    head = csvreader.__next__()
    for row in csvreader:
        y_train_raw.append(row[1])
csvfile.close()
feature_scaler = StandardScaler()
# HOG_x_train = feature_scaler.fit_transform(HOG_x_train)
# HOG_x_test= feature_scaler.transform(HOG_x_test)


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
X_train1, X_val, y_train1, y_val = train_test_split(SOBEL_x_train, y_train_raw, test_size=0.2, random_state=1)
print("yay")

#rf_class= RandomForestClassifier(n_estimators=200,max_depth=30, random_state=0,class_weight='balanced')
rf_class=svm.SVC()
#print(cross_val_score(rf_class, X_train1, y_train1, scoring='accuracy', cv = 10))
accuracy = cross_val_score(rf_class, X_train1, y_train1, scoring='accuracy').mean() * 100
print("Accuracy of Random Forests is: " , accuracy)
if(accuracy>0.26):
    print("YAY")
    rf_class.fit(X_train1, y_train1)
    s=rf_class.score(X_val,y_val)
    print(s)
    if s>0.29:
        #rf_class.fit(HOG_x_train,y_train_raw)
        predicted= rf_class.predict(SOBEL_x_test)


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
#Accuracy:21.6
