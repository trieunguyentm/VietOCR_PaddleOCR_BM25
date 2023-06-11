import argparse
from PIL import Image
import matplotlib.pyplot as plt

from src.tool.predictor import Predictor
from src.tool.config import Cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='path to img file')
    parser.add_argument('--config', required=True, help='name of config file')

    args = parser.parse_args()
    config = Cfg.load_config_from_name(args.config)
    # Load weights từ local cho nhanh, đỡ phải tải
    config['weights'] = './weights/vgg_transformer.pth'
    config['cnn']['pretrained'] = False
    config['device'] = 'cuda:0'

    detector = Predictor(config)

    img = Image.open(args.img)
    s = detector.predict(img)

    print(s)


if __name__ == '__main__':
    main()
