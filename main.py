import argparse
import logging
import cv2
import warnings
import torch
import os
from datasource import create_output_dir
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', type=str,
                        help='channel id')
    parser.add_argument('--video', type=str,
                        help='video path')
    parser.add_argument('--frame_dir', type=str,
                        help='frames dir path')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='cuda device or cpu')
    parser.add_argument('--fps', type=int, default=25,
                        help='cv2.imshow fps')
    parser.add_argument('--show', action='store_true',
                        help='show video using cv2.imshow')
    parser.add_argument('--save', action='store_true',
                        help='save result frames')
    args = parser.parse_args()
    print(f'args: {args}')
    return args


args = parse_args()
out_dir = create_output_dir(args.channel, args.video, args.frame_dir)
logging.getLogger('faiss.loader').setLevel(logging.WARNING)
logging.getLogger('fastreid.utils.checkpoint').setLevel(logging.WARNING)
logging.getLogger('fastreid.engine.defaults').setLevel(logging.WARNING)
logging.getLogger('root.tracker').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
# prod
logging.basicConfig(format='[%(levelname)s] [%(name)s] %(message)s',
                    level=logging.INFO,
                    filename=os.path.join(out_dir, 'log.txt'))
# dev
# logging.basicConfig(format='[%(levelname)s] [%(name)s] %(message)s',
#                     level=logging.DEBUG)
warnings.filterwarnings("ignore", category=UserWarning)

from datasource import get_img_generator
from LPR import single_LPR
from utils import cv2PutChineseText
# from producer_cloud import writeBack

logger = logging.getLogger('main')


def main():
    generator = get_img_generator(args.channel, args.video, args.frame_dir)
    for idx, (img, filename) in enumerate(generator):
        t1 = time.perf_counter()
        plate = single_LPR(img)
        # if plate:
        #     writeBack(plate, False, (0, 0, img.shape[1], img.shape[0]), 'disorder_queue', [filename, filename],
        #               args.channel, 'str2')
        if args.show or args.save:
            frame = draw_frame(img, plate)
        if args.show:
            cv2.imshow('Test', frame)
            cv2.waitKey(1)
        if args.save:
            cv2.imwrite(os.path.join(out_dir, '%07d.jpg' % idx), frame)
        t2 = time.perf_counter()
        logger.debug(f'Time cost per frame: {t2 - t1} s ({1 / (t2 - t1)} FPS)')


def draw_frame(img, plate):
    frame = img.copy()
    if plate:
        frame = cv2PutChineseText(frame, plate, (50, 50), (255, 0, 0), 50)  # 左上角绘制车牌
    return frame


if __name__ == '__main__':
    main()
