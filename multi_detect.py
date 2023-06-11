import requests
import numpy as np
from io import BytesIO
from PIL import Image
import uuid
from src.tool.config import Cfg
from src.tool.predictor import Predictor
import sys
import os
import numpy as np
import yaml
from PIL import Image
from paddleocr import PaddleOCR

def multi_dectect(list_file_paths):
    list_ans = []
    for file_paths in list_file_paths:
        ans = ""
        # Mo anh va chuyen anh thanh ma tran
        im = Image.open(file_paths)
        img = np.asarray(im)
        # Tao doi tuong class PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        # Result la 1 list, chua 1 tensor => kich thuoc (1,x,y,z)
        results = ocr.ocr(img, rec=False, cls=False)
        # boxes chua tat ca toa do bounding box cac text
        boxes = np.asarray([line for line in results[0]])

        print(boxes)
        #TH anh khong co chu
        if(len(boxes) == 0):
            list_ans.append(" ")
            continue
        # sort lai chi so cac box theo thu tu tu tren xuong duoi
        boxes = boxes[boxes[:, 0, 1].argsort()]
        # Tao doi tuong du doan tieng viet
        name_config = "vgg_transformer"
        config = Cfg.load_config_from_name(name_config)
        # Load weights từ local cho nhanh
        config['weights'] = './weights/vgg_transformer.pth'
        config['cnn']['pretrained'] = False
        config['device'] = 'cuda:0'
        detector = Predictor(config)

        img_list = []
        result = []
        box_r = []
        for i, box in enumerate(boxes):
            x_min = int(min(box[:, 0]))
            x_max = int(max(box[:, 0]))
            y_min = int(min(box[:, 1]))
            y_max = int(max(box[:, 1]))
            box_r.append([x_min, y_min, x_max, y_max])
            # Lay anh tu bounding box tach ra duoc
            box_text = img[y_min:y_max, x_min:x_max]
            img_list.append(Image.fromarray(box_text))
            # Set kich thuoc batch ... anh se duoc xu ly cung luc 
            if True: #(i+1) % batch == 0
                sent = detector.predict_batch(img_list)
                img_list = []
                for j, s in enumerate(sent):
                    # result.append({"text": s, "box": box_r[j]})
                    ans += (s + " ")
                # Reset box_r
                box_r = []
        # print(result)
        print("--------------------------------")
        print(ans)
        list_ans.append(ans)
    return list_ans

