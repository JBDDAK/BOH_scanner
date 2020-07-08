from save_img import save_img
import glob
num = 0
for idx, cat in enumerate(["jung"]):
    image_dir = "images" + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        print(f, num)
        save_img(f, num)
        num+=1
