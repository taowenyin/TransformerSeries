import torch
import argparse

from torch.utils.data import DataLoader
from data import *
from utils.augmentations import SSDAugmentation
from utils.vocapi_evaluator import VOCAPIEvaluator


def train():
    args = parse_args()

    # multi-scale
    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = [640, 640]
        val_size = [416, 416]
    else:
        train_size = [416, 416]
        val_size = [416, 416]

    dataset = VOCDetection(root='', img_size=train_size[0], transform=SSDAugmentation(train_size))
    evaluator = VOCAPIEvaluator(data_root='', img_size=val_size[0], device='',
                                transform=BaseTransform(val_size), labelmap=VOC_CLASSES)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=detection_collate, num_workers=args.num_workers, pin_memory=True)
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers used in dataloading')

    return parser.parse_args()


if __name__ == '__main__':
    train()
