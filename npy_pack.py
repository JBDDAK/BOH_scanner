import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
caltech_dir = 'images'
categories = ["jung","rain"]
X = []
Y = []

for idx, cat in enumerate(categories):
    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")

    for i, f in enumerate(files):
        # Y.append(categories.index(cat))
        # X.append(cv2.imread(files[idx]))
        X.append(f)
        Y.append(idx)

X = np.array(X)
Y = np.array(Y)

num = float(input("test_size를 입력해 주세요 (1 전체 테스트 / 0 전체 트레이닝)"))
if(num == 1):#완전한 테스트 케이스
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=1)
    X_test = np.append(X_test, X_train)
    Y_test = np.append(Y_test, Y_train)
elif(num == 0):#완전한 트레이닝
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=1)
    X_train = np.append(X_train, X_test)
    Y_train = np.append(Y_train, Y_test)
else:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=num)

xy = (X_train, X_test, Y_train, Y_test)

np.save("data/all_pack.npy", xy)
print(len(X_train), len(X_test),"끝")