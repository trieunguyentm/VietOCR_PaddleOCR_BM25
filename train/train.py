import argparse

from src.model.trainer import Trainer
from src.tool.config import Cfg


def update_config(config):
    dataset_params = {
        'name': 'hw',
        'data_root': './data_line/',
        'train_annotation': 'train_line_annotation.txt',
        'valid_annotation': 'test_line_annotation.txt'
    }


    params = {
        'batch_size': 4,
        'print_every': 200,
        'valid_every': 15*200,
        'iters': 20000,
        'checkpoint': './checkpoint/transformerocr_checkpoint.pth',
        'export': './weights/transformerocr.pth',
        'metrics': 10000
    }

    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['device'] = 'cuda:0'
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='see example at ')
    parser.add_argument('--checkpoint', required=False, help='your checkpoint')

    args = parser.parse_args()
    config = Cfg.load_config_from_name(args.config)
    config = update_config(config)

    trainer = Trainer(config)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    trainer.train()


if __name__ == '__main__':
    main()
