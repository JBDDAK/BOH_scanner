# 0. 사용할 패키지 불러오기
import numpy as np
import cv2
from save_img import save_img

def img(idx, list):
    for i, f in enumerate(idx):
        img = cv2.imread(f)
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        list.append(img)

def model_learn():
    image_class = ["jung", "rain"]

    test_images = []

    img_model = '../data/test.jpg'
    save_img(img_model, 1)

    img = cv2.imread('../data/QWER.jpg')
    print(img)
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    test_images.append(img)
    #
    test_images = np.array(test_images)

    test_images = test_images.reshape((len(test_images), 256, 256, 3))
    test_images = test_images / 255

    from tensorflow.keras.models import load_model
    model = load_model('../data/all_pack_40.h5')

    # predictions = model.predict(model)
    predictions_test = model.predict(test_images)

    for i in range(len(test_images)):  # 결과 예측값 저장
        a = cv2.imread('../data/QWER.jpg')
        cv2.imwrite('data/{}_{:.2F}%.jpg'.format(image_class[np.argmax(predictions_test[i])],
                                                     100 * np.max(predictions_test[i])), a)
        print('data/{}_{:.2F}%.jpg'.format(image_class[np.argmax(predictions_test[i])],
                                           100 * np.max(predictions_test[i])))
        print(predictions_test, np.argmax(predictions_test[i]), 100 * np.max(predictions_test[i]))
    if(image_class[np.argmax(predictions_test[i])] == "jung"):
        a = "정형돈"
    else:
        a = "비"
    b = 100 * np.max(predictions_test[i])
    return '{} {:.2F}%'.format(a, b)