__author__ = 'takooctopus'

import os
import sys
import cv2
import glob as gb

import time
import json
import re
import pprint
from .dagm import DAGM
import numpy as np
from tqdm import tqdm


class DAGMjson:
    def __init__(self, datasetname="traindagm2007"):
        # super(DAGMjson, self).__init__()
        self.dataDir = '../images/' + datasetname + '/'
        self.cats = os.listdir(self.dataDir)
        self.annotation_file_path = '../annotations/' + datasetname + '.json'
        assert os.path.exists(self.dataDir)

        self._json_configs = {}
        self._configs = {}

        self._json_configs["info"] = {
            "description": "DAGM 2007 dataset",
            "url": "https://resources.mpi-inf.mpg.de/conference/dagm/2007/prizes.html",
            "version": "1.0",
            "year": 2007,
            "contributor": " German Chapter of the European Neural Network Societ",
            "date_created": "2007/08/01"
        }
        self._json_configs["images"] = []
        self._json_configs['licenses'] = []
        self._json_configs['categories'] = []
        self._json_configs['annotations'] = []
        self._configs['license'] = {
            "url": None,
            "id": None,
            "name": None
        }
        self._configs['image'] = {
            "license": 1,
            "file_name": None,
            "dagm_url": None,
            "height": None,
            "width": None,
            "date_captured": None,
            "id": None
        }
        self._configs['category'] = {
            "id": None,
            "name": None,
            "supercategory": "defect",
        }
        self._configs['annotation'] = {
            "id": int,
            "image_id": int,
            "category_id": int,
            # x,y 坐标点集合
            "segmentation": [],
            "area": float,
            "bbox": [],
            "iscrowd": 0,
        }

        self._json_configs['licenses'] = [{
            "url": "https://takooctopus.github.io",
            "id": 1,
            "name": "no license"
        }]

    @property
    def configs(self):
        return self._configs

    @property
    def json_configs(self):
        return self._json_configs

    def read_img(self):
        pass

    def img_to_json(self):
        print("\033[0;33m " + "现在位置:{}/{}/.{}".format(os.getcwd(), os.path.basename(__file__),
                                                      sys._getframe().f_code.co_name) + "\033[0m")
        print("\033[0;36m " + "正在生成标签文件:" + "\033[0m" + "{}".format(self.annotation_file_path))

        """
        将dagm的图像和标签格式化，做成我们需要使用的标签
        :return: 返回在上面的annotation文件夹中
        :return:
        """
        json_configs = self.json_configs
        img_index = 0
        label_index = 0
        img_search_index = 0

        for ind_cat, cat in enumerate(self.cats):
            print("\033[0;33m " + "正在生成第[{}]类标签文件:".format(str(ind_cat + 1)) + "\033[0m" + "{}".format(cat))
            configs = self.configs
            category_config = configs['category']
            category_config.update({
                "id": ind_cat + 1,
                "name": cat,
            })
            if category_config not in json_configs['categories']:
                json_configs['categories'].append(category_config.copy())

            image_dir = self.dataDir + cat + '/'
            label_dir = image_dir + 'Label/'
            img_path = gb.glob(image_dir + "*.PNG")
            label_path = gb.glob(label_dir + "*.PNG")
            for ind_img, path in enumerate(img_path):
                img_index += 1
                image_config = configs['image']
                img = cv2.imread(path)
                shape = img.shape
                # filePath = unicode(path, 'utf8')
                time_stamp = os.path.getmtime(path)
                time_struct = time.localtime(time_stamp)
                file_create_time = time.strftime('%Y-%m-%d %H:%M:%S', time_struct)
                image_config.update({
                    'file_name': os.path.basename(path),
                    'dagm_url': path,
                    'height': shape[0],
                    'width': shape[1],
                    'date_captured': file_create_time,
                    'id': img_index
                })
                if image_config not in json_configs['images']:
                    json_configs['images'].append(image_config.copy())
                # print(json.dumps(image_config, sort_keys=True, indent=4))
                # break

            for ind_img, path in enumerate(label_path):
                label_index += 1
                annotation_config = configs['annotation']
                label = cv2.imread(path)
                label_name = os.path.basename(path)
                img_name = re.sub(r'_label', "", label_name)
                img_id = None
                for i in range(img_search_index, len(json_configs['images'])):
                    if json_configs['images'][i]['file_name'] == img_name:
                        img_id = json_configs['images'][i]['id']
                        # print(i)
                        # print(img_id)
                        # print(json_configs['images'][i]['file_name'])
                        # print(img_name)
                        break
                img_search_index = + len(json_configs['images'])
                img_search_index = - 1
                assert img_id is not None

                gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                segmentation = contours[0].reshape(-1).tolist()
                x, y, w, h = cv2.boundingRect(contours[0])
                area = float(w * h)

                # cv2.rectangle(label, (x, y), (x + w, y + h), (0, 0, 200), 2)
                #
                # while (1):
                #     cv2.imshow('img', label)
                #     cv2.imshow('gray', gray)
                #     cv2.imshow('blur', blur)
                #     cv2.imshow('thresh', thresh)
                #     if cv2.waitKey(1) == ord('q'):
                #         break
                # cv2.destroyAllWindows()

                annotation_config.update({
                    "id": label_index,
                    "image_id": img_id,
                    "category_id": ind_cat + 1,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": [x, y, w, h],
                })
                if annotation_config not in json_configs['annotations']:
                    json_configs['annotations'].append(annotation_config.copy())

        with open(self.annotation_file_path, 'w') as f:
            json.dump(json_configs, f)

    def label_to_json(self):
        print("\033[0;33m " + "现在位置:{}/{}/.{}".format(os.getcwd(), os.path.basename(__file__),
                                                      sys._getframe().f_code.co_name) + "\033[0m")
        print("\033[0;31m " + "正在根据label生成标签文件:" + "\033[0m" + "{}".format(self.annotation_file_path))
        """
        将dagm的图像和标签格式化，做成我们需要使用的标签
        :return: 返回在上面的annotation文件夹中
        :return:
        """
        json_configs = self.json_configs
        img_index = 0
        label_index = 0
        img_search_index = 0

        for ind_cat, cat in enumerate(self.cats):
            print("\033[0;33m " + "正在生成第[{}]类标签文件:".format(str(ind_cat + 1)) + "\033[0m" + "{}".format(cat))
            configs = self.configs
            category_config = configs['category']
            category_config.update({
                "id": ind_cat + 1,
                "name": cat,
            })
            if category_config not in json_configs['categories']:
                json_configs['categories'].append(category_config.copy())

            image_dir = self.dataDir + cat + '/'
            label_dir = image_dir + 'Label/'
            img_paths = gb.glob(image_dir + "*.PNG")
            label_paths = gb.glob(label_dir + "*.PNG")

            for ind_img, path in enumerate(label_paths):
                img_index += 1
                label_index += 1
                image_config = configs['image']
                annotation_config = configs['annotation']
                label_name = os.path.basename(path)
                img_name = re.sub(r'_label', "", label_name)
                img_path = os.path.join(image_dir, img_name)
                img = cv2.imread(img_path)
                label = cv2.imread(path)

                shape = img.shape
                time_stamp = os.path.getmtime(path)
                time_struct = time.localtime(time_stamp)
                file_create_time = time.strftime('%Y-%m-%d %H:%M:%S', time_struct)
                image_config.update({
                    'file_name': os.path.basename(img_path),
                    'dagm_url': img_path,
                    'height': shape[0],
                    'width': shape[1],
                    'date_captured': file_create_time,
                    'id': img_index
                })
                if image_config not in json_configs['images']:
                    json_configs['images'].append(image_config.copy())

                gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                segmentation = contours[0].reshape(-1).tolist()

                left_top_x = shape[0]
                left_top_y = shape[1]
                right_bot_x = 0
                right_bot_y = 0

                for i in range(0, len(segmentation) - 1):
                    # print("\033[0;31m " + "循环[{}]:".format(i) + "\033[0m" + "{}".format(segmentation[i]))
                    if i % 2 == 0:
                        if left_top_x > segmentation[i]:
                            # print("{}----x1---->{}".format(left_top_x, segmentation[i]))
                            left_top_x = segmentation[i]
                        if right_bot_x < segmentation[i]:
                            # print("{}----x2---->{}".format(right_bot_x, segmentation[i]))
                            right_bot_x = segmentation[i]
                    else:
                        if left_top_y > segmentation[i]:
                            # print("{}----y1---->{}".format(left_top_y, segmentation[i]))
                            left_top_y = segmentation[i]
                        if right_bot_y < segmentation[i]:
                            # print("{}----y2---->{}".format(right_bot_y, segmentation[i]))
                            right_bot_y = segmentation[i]

                print("\033[0;36m " + "左上角点:" + "\033[0m" + "{}".format((left_top_x, left_top_y)))
                print("\033[0;36m " + "右下角点:" + "\033[0m" + "{}".format((right_bot_x, right_bot_y)))

                x, y, w, h = left_top_x, left_top_y, right_bot_x - left_top_x, right_bot_y - left_top_y
                area = float(w * h)

                # cv2.rectangle(label, (x, y), (x + w, y + h), (0, 0, 200), 2)
                #
                # while (1):
                #     cv2.imshow('img', label)
                #     cv2.imshow('gray', gray)
                #     cv2.imshow('blur', blur)
                #     cv2.imshow('thresh', thresh)
                #     if cv2.waitKey(1) == ord('q'):
                #         break
                # cv2.destroyAllWindows()

                annotation_config.update({
                    "id": label_index,
                    "image_id": img_index,
                    "category_id": ind_cat + 1,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": [x, y, w, h],
                })
                if annotation_config not in json_configs['annotations']:
                    json_configs['annotations'].append(annotation_config.copy())

        with open(self.annotation_file_path, 'w') as f:
            json.dump(json_configs, f)

    def make_dirs(directories):
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def draw_bboxes(self, image, bboxes, font_size=0.5, thresh=0.5, colors=None):
        print("\033[4;32m " + "现在位置:{}/{}/.{}".format(os.getcwd(), os.path.basename(__file__),
                                                      sys._getframe().f_code.co_name) + "\033[0m")
        """Draws bounding boxes on an image.

        Args:
            image: An image in OpenCV format
            bboxes: A dictionary representing bounding boxes of different object
                categories, where the keys are the names of the categories and the
                values are the bounding boxes. The bounding boxes of category should be
                stored in a 2D NumPy array, where each row is a bounding box (x1, y1,
                x2, y2, score).
            font_size: (Optional) Font size of the category names.
            thresh: (Optional) Only bounding boxes with scores above the threshold
                will be drawn.
            colors: (Optional) Color of bounding boxes for each category. If it is
                not provided, this function will use random color for each category.

        Returns:
            An image with bounding boxes.
        """

        image = image.copy()
        for cat_name in bboxes:
            # 只要大于阈值就行
            keep_inds = bboxes[cat_name][:, -1] > thresh
            print("\033[0;36m " + "类别为:[{}]的bbox经过阈值处理后是否保留:".format(cat_name) + "\033[0m" + "{}".format(keep_inds))
            # 这边是类型的尺寸
            cat_size = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]

            # 创建随机颜色
            if colors is None:
                color = np.random.random((3,)) * 0.6 + 0.4
                color = (color * 255).astype(np.int32).tolist()
            else:
                color = colors[cat_name]

            for bbox in bboxes[cat_name][keep_inds]:
                bbox = bbox[0:4].astype(np.int32)
                if bbox[1] - cat_size[1] - 2 < 0:
                    cv2.rectangle(image,
                                  (bbox[0], bbox[1] + 2),
                                  (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                                  color, -1
                                  )
                    cv2.putText(image, cat_name,
                                (bbox[0], bbox[1] + cat_size[1] + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
                                )
                else:
                    cv2.rectangle(image,
                                  (bbox[0], bbox[1] - cat_size[1] - 2),
                                  (bbox[0] + cat_size[0], bbox[1] - 2),
                                  color, -1
                                  )
                    cv2.putText(image, cat_name,
                                (bbox[0], bbox[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
                                )
                cv2.rectangle(image,
                              (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              color, 2
                              )
        return image

    def test_json(self, split):
        print("\033[0;33m " + "现在位置:{}/{}/.{}".format(os.getcwd(), os.path.basename(__file__),
                                                      sys._getframe().f_code.co_name) + "\033[0m")
        result_dir = "../results"
        result_dir = os.path.join(result_dir, split)
        print("\033[0;36m " + "正在生成测试文件夹:" + "\033[0m" + "{}".format(result_dir))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        debug_dir = os.path.join(result_dir, "debug")
        print("\033[0;36m " + "正在生成debug文件夹:" + "\033[0m" + "{}".format(debug_dir))
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        anno_file = os.path.join("..", "annotations", "{}.json".format(split))
        db = DAGM(anno_file)
        pprint.pprint(db.imgs)
        print("\033[0;36m " + "数据集信息:" + "\033[0m" + "{}".format("(来自takooctopus的转换)"))
        print(db.info())

        db_inds = np.arange(len(db.anns))
        db_inds = db_inds[:1000]
        print("\033[0;36m " + "db_inds(我们取的测试数据集数量index):" + "\033[0m" + "{}".format("看下面啦!!!"))
        print(db_inds)
        num_images = db_inds.size
        categories = len(db.cats)
        print("\033[0;36m " + "num_images(取了多少图片):" + "\033[0m" + "{}".format(num_images))
        print("\033[0;36m " + "categories(种类):" + "\033[0m" + "{}".format(categories))

        print(db.catToImgs)
        print(db.cats)
        print(db.imgToAnns)
        print("\n")

        for ind in tqdm(range(1, num_images + 1), ncols=80, desc="依次循环画出原始bbox看是否转换正确"):
            print("\033[0;36m " + "正在生成bbox:" + "\033[0m")
            anns_id = ind
            ann = db.anns[anns_id]
            print("\033[0;31m " + "读取第[{}]个标签:".format(anns_id) + "\033[0m" + "{}".format(" "))
            pprint.pprint(ann)
            image_id = ann['image_id']
            img = db.imgs[image_id]
            image_name = img['file_name']
            print("\033[0;31m " + "读取对应的图片:" + "\033[0m" + "{}".format(image_name))
            pprint.pprint(img)
            image_path = img['dagm_url']
            print("\033[0;31m " + "使用图片:" + "\033[0m" + "{}".format(image_path))
            image = cv2.imread(image_path)

            cat_id = ann['category_id']
            cat_name = db.cats[cat_id]['name']
            bbox = ann['bbox']
            bbox.append(1)
            bboxes = {
                cat_name: np.array([bbox])
            }
            print("\033[0;31m " + "使用边框:" + "\033[0m" + "{}".format(bboxes))
            print("\033[0;36m " + "正在生成边框..." + "\033[0m")
            image = self.draw_bboxes(image, bboxes, thresh=0.5)

            segmentation = ann['segmentation']
            print("\033[0;31m " + "segmentation:" + "\033[0m" + "{}".format(segmentation))

            for i in range(0, len(segmentation) - 1, 2):
                point = (segmentation[i], segmentation[i + 1])
                cv2.circle(image, point, 1, (0, 0, 255), 4)

            debug_file = os.path.join(debug_dir, "{}.jpg".format(ind))
            print("\033[0;36m " + "正在输出图片(带bbox):" + "\033[0m" + "{}".format(debug_file))
            cv2.imwrite(debug_file, image)
            # breakpoint()
