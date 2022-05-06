# -*- coding: utf-8 -*-
# __author__:Livingbody
# 2022/5/6 19:48
import paddlehub as hub
import cv2
import os


# 数据mask
def mosaic(selected_image, nsize=9):
    rows, cols, _ = selected_image.shape
    dist = selected_image.copy()
    # 划分小方块，每个小方块填充随机颜色
    for y in range(0, rows, nsize):
        for x in range(0, cols, nsize):
            # 随机
            # dist[y:y + nsize, x:x + nsize] = (np.random.randint(0, 255))
            # 用255白色mask
            dist[y:y + nsize, x:x + nsize] = 255
    return dist


# source_dir: 待脱敏图片文件夹
# target_dir: 脱敏后图片保存文件夹
# key_words： 检测盒的基本信息（不用mask掉文字列表)
def data_mask(source_dir, target_dir, key_words=['2019', 'ANC', "C", "T", 'S', "c", "t", "s"]):
    ocr = hub.Module(name="chinese_ocr_db_crnn_server")

    for root, dirs, files in os.walk(source_dir):
        test_img_path = files
        test_img_path = [os.path.join(source_dir, file) for file in test_img_path]
    # print("test_img_path", test_img_path)
    # 读取测试文件夹test.txt中的照片路径
    np_images = [cv2.imread(image_path, 1) for image_path in test_img_path]

    results = ocr.recognize_text(
        images=np_images,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
        use_gpu=False,  # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
        output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
        visualization=False,  # 是否将识别结果保存为图片文件；
        box_thresh=0.5,  # 检测文本框置信度的阈值；
        text_thresh=0.3)  # 识别中文文本置信度的阈值；
    print("results:", results)
    for i in range(len(results)):
        result = results[i]
        data = result['data']
        print(f"{i}*******************************************")
        for infomation in data:
            # print('text: ', infomation['text'], '\nconfidence: ', infomation['confidence'], '\ntext_box_position: ',
            #       infomation['text_box_position'])
            # print(infomation['text'])
            flag = True
            for word in key_words:
                if word in infomation['text']:
                    flag = False
                    break
            if flag == True:
                cut_point = infomation['text_box_position']
                roiImg = np_images[i][cut_point[0][1]:cut_point[2][1],
                         cut_point[0][0]:cut_point[2][0]]  # 使用数组切片的方式截取载入图片上的部分,
                mosaic_result = mosaic(roiImg)
                np_images[i][cut_point[0][1]:cut_point[2][1],
                cut_point[0][0]:cut_point[2][0]] = mosaic_result  # 然后,将截取的这部分ROI区域的图片保存在roiImg矩阵变量中
        # print(f"{i}*******************************************")
        cv2.imwrite(filename=os.path.join(target_dir, f"{i}.jpg"), img=np_images[i])
