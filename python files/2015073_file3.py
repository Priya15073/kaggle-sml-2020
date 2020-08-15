import cv2
import csv
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
TRAIN_FOLDER = "/Users/priyarajpurohit/Desktop/bonus-sml-2020/SML_Train/Train_"
TRAIN_LABEL = "/Users/priyarajpurohit/Desktop/bonus-sml-2020/SML_Train.csv"
TEST_FOLDER = "/Users/priyarajpurohit/Desktop/bonus-sml-2020/SML_Test/Test_"
TEST_LABEL = "/Users/priyarajpurohit/Desktop/bonus-sml-2020/2015073_Priya_submission.csv"
test_label="/Users/priyarajpurohit/Desktop/bonus-sml-2020/2015073_model3.csv"
model_label="2015073_model3.pkl"
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

dt = DecisionTreeClassifier()
dt.fit(X_train_raw,y_train_raw)
predicted= dt.predict(X_test_raw)
pickle.dump(dt, open(model_label, 'wb'))
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
