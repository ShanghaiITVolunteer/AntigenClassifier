import os

#先写class_label.txt文件
class_label = open("labels.txt", "w")

for class_name in os.listdir("gallery"):
    if (class_name == "others") or (class_name == "item_label.txt"):
        continue
    if len(class_name.strip().split(" ")) != 1:
        continue
    for img_path in os.listdir("gallery/{}".format(class_name)):
        if len(img_path.strip().split(" ")) != 1:
            continue
        #print(img_path)
        class_label.write("{}/{}\t{}\n".format(class_name, img_path, class_name))