import cv2
import csv
import pickle
import numpy as np
from skimage.io import imread, imshow
import sklearn.preprocessing as prp
from skimage.restoration import  denoise_bilateral
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
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
TEST_LABEL = "/Users/priyarajpurohit/Desktop/bonus-sml-2020/2015073_Priya_submission.csv"
test_label="/Users/priyarajpurohit/Desktop/bonus-sml-2020/2015073_model14.csv"
model_label="2015073_model14.pkl"
hog_train = []
hsv_train=[]

rgb_train=[]
rgb_test=[]
y_train_raw = []
hog_test = []

hsv_test=[]

for i in range(16000):
    f = TRAIN_FOLDER + str(i) + ".jpg"
    im=cv2.imread(f)

    im2=  cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    fd = hog(im, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(4, 4),block_norm= 'L2')
    hog_train.append(fd)

    hsv_train.append(im2.flatten())
    rgb_train.append(im.flatten())
for i in range(1500):
    f = TEST_FOLDER + str(i) + ".jpg"
    im=cv2.imread(f)

    im2=  cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    fd = hog(im, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(4, 4),block_norm= 'L2')
    hog_test.append(fd)

    hsv_test.append(im2.flatten())
    rgb_test.append(im.flatten())

with open(TRAIN_LABEL, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    head = csvreader.__next__()
    for row in csvreader:
        y_train_raw.append(row[1])
csvfile.close()

hsv_train=prp.quantile_transform(hsv_train)
hsv_test=prp.quantile_transform(hsv_test)
hog_train=prp.quantile_transform(hog_train)
hog_test=prp.quantile_transform(hog_test)
rgb_train=prp.quantile_transform(rgb_train)
rgb_test=prp.quantile_transform(rgb_test)

pca = PCA(n_components=50).fit(hsv_train)
hsv_train = pca.transform(hsv_train)
hsv_test = pca.transform(hsv_test)

pca = PCA(n_components=50).fit(hog_train)
hog_train = pca.transform(hog_train)
hog_test = pca.transform(hog_test)

pca = PCA(n_components=50).fit(rgb_train)
rgb_train = pca.transform(rgb_train)
rgb_test = pca.transform(rgb_test)

X_train_raw=np.concatenate((np.array(hog_train),np.array(rgb_train),np.array(hsv_train)),axis=1)
X_test_raw=np.concatenate((np.array(hog_test),np.array(rgb_test),np.array(hsv_test)),axis=1)
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

rf_class= RandomForestClassifier(n_estimators=500,max_depth=30, random_state=42,class_weight='balanced')
#rf_class=svm.SVC()
#print(cross_val_score(rf_class, X_train1, y_train1, scoring='accuracy', cv = 10))
accuracy = cross_val_score(rf_class, X_train1, y_train1, scoring='accuracy').mean() * 100
print("Accuracy of Random Forests is: " , accuracy)
if(accuracy>0.32):
    print("YAY")
    rf_class.fit(X_train1, y_train1)
    s=rf_class.score(X_val,y_val)
    print(s)
    if s>0.36:
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
#0.3790625
