__author__ = 'takooctopus'

import os
import cv2
import glob as gb

import time
import json
import re


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
    def jsonConfigs(self):
        return self._json_configs

    def readImg(self):
        pass

    def imgToJson(self):
        print("正在生成标签文件： " + self.annotation_file_path)
        """
        将dagm的图像和标签格式化，做成我们需要使用的标签
        :return: 返回在上面的annotation文件夹中
        :return:
        """
        json_configs = self.jsonConfigs
        img_index = 0
        label_index = 0
        img_search_index = 0

        for ind_cat, cat in enumerate(self.cats):
            print(str(ind_cat + 1) + " : " + cat)
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
