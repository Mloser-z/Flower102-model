import numpy as np
import os
from .classify import Classifier
from PIL import Image


class ImageMatcher:
    def __init__(self, model_path: str, dataset_path):
        # self.reference_image = cv2.imread(reference_image)
        self.dataset_path = dataset_path
        # self.CLASSES = 102
        self.classifier = Classifier(model_path)

    def difference_hash(self, img, width=9, length=8):
        img = img.resize((width, length)).convert('L')
        img_matrix = np.array(img)
        # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
        hash = []

        for i in range(length):
            for j in range(1, width):
                if img_matrix[i, j] > img_matrix[i, j - 1]:
                    hash.append(1)
                else:
                    hash.append(0)
        return hash

    def get_hamming_distance(self, hash1, hash2):
        distance = 0
        for index in range(len(hash1)):
            if hash1[index] != hash2[index]:
                distance += 1
        return distance

    """ 返回相似图片 [hamming distance, 绝对路径] 列表。
    category默认为0，全局搜索；topk为返回的图片数。"""

    def get_similar_images(self, reference, category, topk=2):
        search_dir = str(category) + "/"
        images_path_list = os.listdir(self.dataset_path + search_dir)
        similar_img_list = []

        # start = time.time()
        for img in images_path_list:
            path = search_dir + img
            i = Image.open(self.dataset_path + path)
            distance = self.get_hamming_distance(self.difference_hash(reference),
                                                 self.difference_hash(i))
            if len(similar_img_list) < topk:
                similar_img_list.append([distance, path])
                similar_img_list.sort()

            elif similar_img_list[-1][0] > distance:
                del similar_img_list[-1]
                similar_img_list.append([distance, path])
                similar_img_list.sort()
        for i in range(topk):
            similar_img_list[i] = similar_img_list[i][1]
        # end = time.time()
        # print("searching time = "+str(end - start))
        return similar_img_list

    ''' 分类图片，并返回预测的5种类别、概率、相似图片
        img_path 是待分类图片的路径。'''

    def classify_and_search_similar_images(self, img_path):
        """ 调用classify.py 分类图片 """
        topk_categories, topk_prob = self.classifier.classify(img_path)
        reference = Image.open(img_path)

        """ 查找相似图片 """
        classify_list = []
        # 按照分类查找本类最相似的2张图片
        for i in range(len(topk_categories)):
            similar_img_list = self.get_similar_images(
                reference, category=topk_categories[i])
            label_prob_path = [topk_categories[i],
                               topk_prob[i], similar_img_list]
            classify_list.append(label_prob_path)
        return classify_list
