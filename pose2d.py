import argparse
import os
import platform
import time

import torch
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
from alphapose.utils.writer import DataWriter
from detector.apis import get_detector
from tqdm import tqdm

CFG = 'configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'
CHECKPOINT = 'pretrained_models/halpe26_fast_res50_256x192.pth'


def get_parser():
    parser = argparse.ArgumentParser(description='AlphaPose Demo')
    parser.add_argument('--cfg', type=str, required=False,
                        help='experiment configure file name')
    parser.add_argument('--checkpoint', type=str, required=False,
                        help='checkpoint file name')
    parser.add_argument('--sp', default=False, action='store_true',
                        help='Use single process for pytorch')
    parser.add_argument('--detector', dest='detector',
                        help='detector name', default="yolo")
    parser.add_argument('--detfile', dest='detfile',
                        help='detection result file', default="")
    parser.add_argument('--indir', dest='inputpath',
                        help='image-directory', default="")
    parser.add_argument('--list', dest='inputlist',
                        help='image-list', default="")
    parser.add_argument('--image', dest='inputimg',
                        help='image-name', default="")
    parser.add_argument('--outdir', dest='outputpath',
                        help='output-directory', default="examples/res/")
    parser.add_argument('--save_img', default=False, action='store_true',
                        help='save result as image')
    parser.add_argument('--vis', default=False, action='store_true',
                        help='visualize image')
    parser.add_argument('--showbox', default=False, action='store_true',
                        help='visualize human bbox')
    parser.add_argument('--profile', default=False, action='store_true',
                        help='add speed profiling at screen output')
    parser.add_argument('--format', type=str,
                        help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
    parser.add_argument('--min_box_area', type=int, default=0,
                        help='min box area to filter out')
    parser.add_argument('--detbatch', type=int, default=5,
                        help='detection batch size PER GPU')
    parser.add_argument('--posebatch', type=int, default=64,
                        help='pose estimation maximum batch size PER GPU')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                        help='save the result json as coco format, using image index(int) instead of image name(str)')
    parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                        help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
    parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                        help='the length of result buffer, where reducing it will lower requirement of cpu memory')
    parser.add_argument('--flip', default=False, action='store_true',
                        help='enable flip testing')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='print detail information')
    """----------------------------- Video options -----------------------------"""
    parser.add_argument('--video', dest='video',
                        help='video-name', default="")
    parser.add_argument('--webcam', dest='webcam', type=int,
                        help='webcam number', default=-1)
    parser.add_argument('--save_video', dest='save_video',
                        help='whether to save rendered video', default=False, action='store_true')
    parser.add_argument('--vis_fast', dest='vis_fast',
                        help='use fast rendering', action='store_true', default=False)
    """----------------------------- Tracking options -----------------------------"""
    parser.add_argument('--pose_flow', dest='pose_flow',
                        help='track humans in video with PoseFlow', action='store_true', default=False)
    parser.add_argument('--pose_track', dest='pose_track',
                        help='track humans in video with reid', action='store_true', default=False)

    args = parser.parse_args()

    return args


def print_finish_info(args):
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print(
            '===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')


def loop():
    n = 0
    while True:
        yield n
        n += 1


def parse(video_name):
    args = get_parser()
    cfg = update_config(CFG)

    if platform.system() == 'Windows':
        args.sp = True

    args.video = video_name
    args.cfg = CFG
    args.checkpoint = CHECKPOINT
    args.device = 'cuda'
    args.outputpath = './output'
    args.gpus = [0]
    args.detbatch = args.detbatch * len(args.gpus)
    args.posebatch = args.posebatch * len(args.gpus)
    args.tracking = args.pose_track or args.pose_flow or args.detector == 'tracker'

    det_loader = DetectionLoader(video_name, get_detector(args), cfg, args, batchSize=args.detbatch, mode='video',
                                 queueSize=args.qsize)
    det_worker = det_loader.start()

    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    pose_model.to(args.device)
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    queue_size = args.qsize
    video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_' + os.path.basename(video_name))
    video_save_opt.update(det_loader.videoinfo)
    writer = DataWriter(cfg, args, save_video=True, video_save_opt=video_save_opt, queueSize=queue_size).start()

    data_len = det_loader.length
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
    batch_size = args.posebatch

    try:
        for _ in im_names_desc:
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue
                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if datalen % batch_size:
                    leftover = 1
                num_batches = datalen // batch_size + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batch_size:min((j + 1) * batch_size, datalen)]
                    if args.flip:
                        inps_j = torch.cat((inps_j, flip(inps_j)))
                    hm_j = pose_model(inps_j)
                    if args.flip:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                        hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)
                hm = torch.cat(hm)
                hm = hm.cpu()
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)

        print_finish_info(args)
        while writer.running():
            time.sleep(1)
            print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...',
                  end='\r')
        writer.stop()
        det_loader.stop()
    except Exception as e:
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')
        pass
    except KeyboardInterrupt:
        print_finish_info(args)
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            det_loader.terminate()
            while writer.running():
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(
                    writer.count()) + ' images in the queue...', end='\r')
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues

            det_loader.terminate()
            writer.terminate()
            writer.clear_queues()
            det_loader.clear_queues()


if __name__ == '__main__':
    video_name = './videos/kunkun_cut.mp4'
    json_file = parse(video_name)
    # TODO: where is json?
