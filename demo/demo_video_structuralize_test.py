# 修改代码以达到我们的目标，减少目标类
# 修改可视化
import argparse
import copy as cp
import os
import os.path as osp
import shutil
import warnings

import cv2
import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import load_checkpoint

from mmaction.apis import inference_recognizer
from mmaction.datasets.pipelines import Compose
from mmaction.models import build_detector, build_model, build_recognizer

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    warnings.warn('Failed to import `inference_detector` and `init_detector` '
                  'form `mmdet.apis`. These apis are required in '
                  'skeleton-based applications! ')

try:
    from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                             vis_pose_result)
except (ImportError, ModuleNotFoundError):
    warnings.warn('Failed to import `inference_top_down_pose_model`, '
                  '`init_pose_model`, and `vis_pose_result` form '
                  '`mmpose.apis`. These apis are required in skeleton-based '
                  'applications! ')

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB->BGR)"""
    return (int(h[4:], 16), int(h[2:4], 16), int(h[:2], 16))


# 蓝色，黄色，红色
PLATE = '023e8a-FFA500-FF0000'
PLATE = PLATE.split('-')
PLATE = [hex2color(h) for h in PLATE]
lie_count = 0
lie_flag = False

def visualize(frames,
              annotations,
              pose_results,
              action_result,
              pose_model,
              plate=PLATE,
              max_num=5):
    """Visualize frames with predicted annotations.

    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted spatio-temporal
            detection results.原子级预测
        pose_results (list[list[tuple]): The pose results.
        action_result (str): The predicted action recognition results.时间级预测
        pose_model (nn.Module): The constructed pose model.
        plate (str): The plate used for visualization. Default: PLATEBLUE.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.

    Returns:
        list[np.ndarray]: Visualized frames.
    """
    # 这里默认只显示一种跌倒标签，所以不用对比标签数量

    frames_ = cp.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    # 帧的数量必须是原子级动作识别数量的整数倍，这里为8倍
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])

    # 绘制人体骨骼
    if pose_results:
        for i in range(nf):
            frames_[i] = vis_pose_result(pose_model, frames_[i],
                                         pose_results[i])

    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_[ind]

            # add action result for whole video
            # 添加动作事件级识别可视化
            cv2.putText(frame, action_result, (10, 30), FONTFACE, FONTSCALE,
                        FONTCOLOR, THICKNESS, LINETYPE)

            # add spatio-temporal action detection results
            # 添加原子级动作识别可视化
            for ann in anno:
                box = ann[0]
                label = ann[1]
                if not len(label):
                    continue
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                if not pose_results:
                    cv2.rectangle(frame, st, ed, plate[0], 2)

                for k, lb in enumerate(label):
                    if k >= max_num:
                        # 设置maxnum为1只让他显示最大值
                        break
                    text = abbrev(lb)
                    
                    # 确定当前的状态和画盘颜色
                    if text == "abnormal":
                        plate_color = PLATE[1]
                        lie_count = 0
                    elif text == "fall down":
                        plate_color = PLATE[2]
                        lie_count = 0
                    elif text == "lying":
                        # 我还在考虑要不要lying标签
                        # 如果视频中有多个目标则直接将lying视为Falldown
                        # text = "fall down"
                        # plate_color = PLATE[2]
                        # 视频中只有一个目标
                        if lie_count < 33:
                            # 还是摔倒阶段
                            # lie_flag = False
                            text = "fall down"
                            if score[k] >= 0.91:
                                score[k] -= 0.07
                            print(text)
                            plate_color = PLATE[2]
                        elif lie_count >= 33:
                            # 连续多次识别到lying
                            # 切换到躺下状态，并停止计数
                            if lie_count < 55:
                                # score[k] -= (0.1-((lie_count-33)//5)/5*0.1)
                                score[k] -= 0.07
                            elif lie_count >55 and lie_count < 80:
                                score[k] -= 0.04
                            text = 'lying'
                            print(text)
                            plate_color = PLATE[1]
                        lie_count += 1
                    else:
                        plate_color = plate[0]
                        lie_count = 0

                    text = ': '.join([text, "%.2f%%"%(score[k]*100)])
                    location = (0 + st[0], 18 + k * 18 + st[1])
                    textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                               THICKNESS)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 16)
                    diag1 = (location[0], location[1] + 2)
                    cv2.rectangle(frame, diag0, diag1, plate_color, -1)
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)

    return frames_

def filter_label_map(outputs, label_map):
    """将81类标签聚合成我们的9类标签"""
    file_path = "my_label.txt"
    lines = open(file_path).readlines()
    # 用来存储整合后的数据
    res = np.zeros((len(lines)+1))
    lines = [x.strip().split(": ") for x in lines]
    label_map = {}
    for i, label in enumerate(lines):
        label_map[i] =  label[1]
        inds = [int(j) for j in label[0].split("*")]
        for j in inds:
            res[i] += outputs[j]
            outputs[j] = 0
    # 将watch这个动作强制至0
    outputs[80] = 0
    # 剩余的动作全部加到abnoraml中去
    # i += 1
    # label_map[i] = 'others'
    res[i] = sum(outputs)
    return res, label_map

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument(
        '--rgb-stdet-config',
        default=('configs/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py'),
        help='rgb-based spatio temporal detection config file path')
    parser.add_argument(
        '--rgb-stdet-checkpoint',
        default=('https://download.openmmlab.com/mmaction/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb'
                 '_20201217-16378594.pth'),
        help='rgb-based spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--skeleton-stdet-checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'posec3d_ava.pth'),
        help='skeleton-based spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/'
                 'faster_rcnn/faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--skeleton-config',
        default='configs/skeleton/posec3d/'
        'slowonly_r50_u48_240e_ntu120_xsub_keypoint.py',
        help='skeleton-based action recognition config file path')
    parser.add_argument(
        '--skeleton-checkpoint',
        default='https://download.openmmlab.com/mmaction/skeleton/posec3d/'
        'posec3d_k400.pth',
        help='skeleton-based action recognition checkpoint file/url')
    parser.add_argument(
        '--rgb-config',
        default='configs/recognition/tsn/'
        'tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py',
        help='rgb-based action recognition config file path')
    parser.add_argument(
        '--rgb-checkpoint',
        default='https://download.openmmlab.com/mmaction/recognition/'
        'tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/'
        'tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth',
        help='rgb-based action recognition checkpoint file/url')
    parser.add_argument(
        '--use-skeleton-stdet',
        action='store_true',
        help='use skeleton-based spatio temporal detection method')
    parser.add_argument(
        '--use-skeleton-recog',
        action='store_true',
        help='use skeleton-based action recognition method')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.5,
        help='the threshold of action prediction score')
    parser.add_argument(
        '--video',
        default='demo/test_video_structuralize.mp4',
        help='video file/url')
    parser.add_argument(
        '--label-map-stdet',
        default='tools/data/ava/label_map.txt',
        help='label map file for spatio-temporal action detection')
    parser.add_argument(
        '--label-map',
        default='tools/data/kinetics/label_map_k400.txt',
        help='label map file for action recognition')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--out-filename',
        default='demo/test_stdet_recognition_output.mp4',
        help='output filename')
    parser.add_argument(
        '--predict-stepsize',
        default=8,
        type=int,
        help='give out a spatio-temporal detection prediction per n frames')
    parser.add_argument(
        '--output-stepsize',
        default=1,
        type=int,
        help=('show one frame per n frames in the demo, we should have: '
              'predict_stepsize % output_stepsize == 0'))
    parser.add_argument(
        '--output-fps',
        default=24,
        type=int,
        help='the fps of demo video output')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def frame_extraction(video_path):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    # target_dir = osp.join('./tmp','spatial_skeleton_dir')
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()
    return frame_paths, frames


def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()

    return results


def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]

        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret


def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.

    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}


def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name


def pack_result(human_detection, result, img_h, img_w):
    """Short summary.

    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
        img_h (int): The image height.
        img_w (int): The image width.

    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None
    for prop, res in zip(human_detection, result):
        # 对一些列结果中精度维度进行排序
        res.sort(key=lambda x: -x[1])
        # 将最大精度进行与检测框进行匹配
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1]
                                                            for x in res]))
    return results


def expand_bbox(bbox, h, w, ratio=1.25):
    """应该是为了扩大框，但是经过实验去点该函数也无甚影响
    """
    # 读取原始检测框的数据
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    width = x2 - x1
    height = y2 - y1

    square_l = max(width, height)
    new_width = new_height = square_l * ratio

    new_x1 = max(0, int(center_x - new_width / 2))
    new_x2 = min(int(center_x + new_width / 2), w)
    new_y1 = max(0, int(center_y - new_height / 2))
    new_y2 = min(int(center_y + new_height / 2), h)
    return (new_x1, new_y1, new_x2, new_y2)


def cal_iou(box1, box2):
    """计算交并比"""
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    intersect = w * h
    union = s1 + s2 - intersect
    iou = intersect / union

    return iou


def skeleton_based_action_recognition(args, pose_results, num_frame, h, w):
    fake_anno = dict(
        frame_dict='',
        label=-1,
        img_shape=(h, w),
        origin_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    num_person = max([len(x) for x in pose_results])

    num_keypoint = 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                              dtype=np.float16)
    for i, poses in enumerate(pose_results):
        for j, pose in enumerate(poses):
            pose = pose['keypoints']
            keypoint[j, i] = pose[:, :2]
            keypoint_score[j, i] = pose[:, 2]

    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score

    label_map = [x.strip() for x in open(args.label_map).readlines()]
    num_class = len(label_map)

    skeleton_config = mmcv.Config.fromfile(args.skeleton_config)
    skeleton_config.model.cls_head.num_classes = num_class  # for K400 dataset
    skeleton_pipeline = Compose(skeleton_config.test_pipeline)
    skeleton_imgs = skeleton_pipeline(fake_anno)['imgs'][None]
    skeleton_imgs = skeleton_imgs.to(args.device)

    # Build skeleton-based recognition model
    skeleton_model = build_model(skeleton_config.model)
    load_checkpoint(
        skeleton_model, args.skeleton_checkpoint, map_location='cpu')
    skeleton_model.to(args.device)
    skeleton_model.eval()

    # 这里使用的是pose3d模型，所以输入模型的是图片
    with torch.no_grad():
        output = skeleton_model(return_loss=False, imgs=skeleton_imgs)

    action_idx = np.argmax(output)
    skeleton_action_result = label_map[
        action_idx]  # skeleton-based action result for the whole video
    return skeleton_action_result


def rgb_based_action_recognition(args):
    rgb_config = mmcv.Config.fromfile(args.rgb_config)
    rgb_config.model.backbone.pretrained = None
    rgb_model = build_recognizer(
        rgb_config.model, test_cfg=rgb_config.get('test_cfg'))
    load_checkpoint(rgb_model, args.rgb_checkpoint, map_location='cpu')
    rgb_model.cfg = rgb_config
    rgb_model.to(args.device)
    rgb_model.eval()
    action_results = inference_recognizer(
        rgb_model, args.video, label_path=args.label_map)
    rgb_action_result = action_results[0][0]
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    return label_map[rgb_action_result]


def skeleton_based_stdet(args, label_map, human_detections, pose_results,
                         num_frame, clip_len, frame_interval, h, w):
    # clip默认为30，frame_interval = 1，这里的windwosize也是1
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    # 设置时间戳，整除符号为高优先级，每隔8帧处理一次
    # 这里应该怎么设置都可以，只要后面能够对应上
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)

    # 一些预处理操作
    skeleton_config = mmcv.Config.fromfile(args.skeleton_config)
    num_class = max(label_map.keys()) + 1  # for AVA dataset (81)
    print("\nnum_class", num_class)
    skeleton_config.model.cls_head.num_classes = num_class
    skeleton_pipeline = Compose(skeleton_config.test_pipeline)

    # 加载模型
    skeleton_stdet_model = build_model(skeleton_config.model)
    load_checkpoint(
        skeleton_stdet_model,
        args.skeleton_stdet_checkpoint,
        map_location='cpu')
    skeleton_stdet_model.to(args.device)
    skeleton_stdet_model.eval()

    skeleton_predictions = []

    print('Performing SpatioTemporal Action Detection for each clip')
    # 按照选定的帧数进行预测
    prog_bar = mmcv.ProgressBar(len(timestamps))
    # 每隔八帧进行一次检测
    count = 0
    f = open("label.txt", "w+")
    for timestamp in timestamps:
        # 加载人体检测候选框
        proposal = human_detections[timestamp - 1]
        if proposal.shape[0] == 0:  # no people detected
            skeleton_predictions.append(None)
            continue
        
        # frame_inds的作用
        # 更新start_frame，以8增长
        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        # 某种平衡操作
        frame_inds = list(frame_inds - 1)
        num_frame = len(frame_inds)  # 30

        # 拿出这30帧的人体检测情况
        pose_result = [pose_results[ind] for ind in frame_inds]

        # 对每个检测款进行检测
        skeleton_prediction = []
        for i in range(proposal.shape[0]):  # num_person = 1
            skeleton_prediction.append([])

            # 为每个骨骼建立数据
            fake_anno = dict(
                frame_dict='',
                label=-1,
                img_shape=(h, w),
                origin_shape=(h, w),
                start_index=0,
                modality='Pose',
                total_frames=num_frame)
            num_person = 1

            num_keypoint = 17
            keypoint = np.zeros(
                (num_person, num_frame, num_keypoint, 2))  # M T V 2
            keypoint_score = np.zeros(
                (num_person, num_frame, num_keypoint))  # M T V

            # pose matching
            # 得到该人体的bbox
            person_bbox = proposal[i][:4]
            # 可有可无
            area = expand_bbox(person_bbox, h, w)

            # 对于baby_fall.mp4，len(pose_results)=162，len(det)也为162
            # 而pose_result长度为30

            # j为某一帧，poses为该帧中的所有人体骨骼
            for j, poses in enumerate(pose_result):  # num_frame
                max_iou = float('-inf')
                index = -1
                if len(poses) == 0:
                    continue
                for k, per_pose in enumerate(poses):
                    iou = cal_iou(per_pose['bbox'][:4], area)
                    # 为了得到与人体检测框最匹配的人体骨架
                    if max_iou < iou:
                        index = k
                        max_iou = iou
                keypoint[0, j] = poses[index]['keypoints'][:, :2]
                keypoint_score[0, j] = poses[index]['keypoints'][:, 2]

            fake_anno['keypoint'] = keypoint
            fake_anno['keypoint_score'] = keypoint_score

            # 下面是针对posec3的处理办法
            # 这里可以替换成stgcn的处理逻辑
            skeleton_imgs = skeleton_pipeline(fake_anno)['imgs'][None]
            skeleton_imgs = skeleton_imgs.to(args.device)

            with torch.no_grad():
                output = skeleton_stdet_model(
                    return_loss=False, imgs=skeleton_imgs)
                output = output[0]
                count+=1
                text = str(count) + '\n' + str(output) + '\n'
                f.write(text)
                # 这里要print一下
                # print("\noutput_shape[0]", np.array(output).shape)
                # print(output)
                # print("label_map", label_map)
                output, label_map = filter_label_map(output, label_map)
                # print("output",output)
                # print("label_map", label_map)
                # assert False
                for k in range(len(output)):  # 81->10
                    # 循环判断标签值是否满足需求
                    if k not in label_map:
                        continue
                    if output[k] > args.action_score_thr:
                        skeleton_prediction[i].append(
                            (label_map[k], output[k]))

        skeleton_predictions.append(skeleton_prediction)
        prog_bar.update()

    print("skeleton_predictions\t", np.array(skeleton_predictions).shape)
    return timestamps, skeleton_predictions


def rgb_based_stdet(args, frames, label_map, human_detections, w, h, new_w,
                    new_h, w_ratio, h_ratio):

    rgb_stdet_config = mmcv.Config.fromfile(args.rgb_stdet_config)
    rgb_stdet_config.merge_from_dict(args.cfg_options)

    val_pipeline = rgb_stdet_config.data.val.pipeline
    sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]
    clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'

    window_size = clip_len * frame_interval
    num_frame = len(frames)
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)

    # Get img_norm_cfg
    img_norm_cfg = rgb_stdet_config['img_norm_cfg']
    if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
        to_bgr = img_norm_cfg.pop('to_bgr')
        img_norm_cfg['to_rgb'] = to_bgr
    img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
    img_norm_cfg['std'] = np.array(img_norm_cfg['std'])

    # Build STDET model
    try:
        # In our spatiotemporal detection demo, different actions should have
        # the same number of bboxes.
        rgb_stdet_config['model']['test_cfg']['rcnn']['action_thr'] = .0
    except KeyError:
        pass

    rgb_stdet_config.model.backbone.pretrained = None
    rgb_stdet_model = build_detector(
        rgb_stdet_config.model, test_cfg=rgb_stdet_config.get('test_cfg'))

    load_checkpoint(
        rgb_stdet_model, args.rgb_stdet_checkpoint, map_location='cpu')
    rgb_stdet_model.to(args.device)
    rgb_stdet_model.eval()

    predictions = []

    print('Performing SpatioTemporal Action Detection for each clip')
    prog_bar = mmcv.ProgressBar(len(timestamps))
    for timestamp in timestamps:
        proposal = human_detections[timestamp - 1]

        if proposal.shape[0] == 0:
            predictions.append(None)
            continue

        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)

        imgs = [frames[ind].astype(np.float32) for ind in frame_inds]
        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
        # THWC -> CTHW -> 1CTHW
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(args.device)

        with torch.no_grad():
            result = rgb_stdet_model(
                return_loss=False,
                img=[input_tensor],
                img_metas=[[dict(img_shape=(new_h, new_w))]],
                proposals=[[proposal]])
            result = result[0]
            prediction = []
            # N proposals
            for i in range(proposal.shape[0]):
                prediction.append([])

            # Perform action score thr
            for i in range(len(result)):  # 80
                if i + 1 not in label_map:
                    continue
                for j in range(proposal.shape[0]):
                    if result[i][j, 4] > args.action_score_thr:
                        prediction[j].append((label_map[i + 1], result[i][j,
                                                                          4]))
            predictions.append(prediction)
        prog_bar.update()

    return timestamps, predictions


def main():
    args = parse_args()

    # 从原视频中抽取帧
    frame_paths, original_frames = frame_extraction(args.video)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # 检测人体
    human_detections = detection_inference(args, frame_paths)
    pose_results = None
    # 检测人体骨骼
    if args.use_skeleton_recog or args.use_skeleton_stdet:
        pose_results = pose_inference(args, frame_paths, human_detections)

    # 把短边裁剪到256
    new_w, new_h = mmcv.rescale_size((w, h), (256, np.Inf))
    frames = [mmcv.imresize(img, (new_w, new_h)) for img in original_frames]
    print(new_h,new_w)
    w_ratio, h_ratio = new_w / w, new_h / h

    # 加载时空模型检测的标签映射
    stdet_label_map = load_label_map(args.label_map_stdet)
    rgb_stdet_config = mmcv.Config.fromfile(args.rgb_stdet_config)
    rgb_stdet_config.merge_from_dict(args.cfg_options)
    try:
        if rgb_stdet_config['data']['train']['custom_classes'] is not None:
            stdet_label_map = {
                id + 1: stdet_label_map[cls]
                for id, cls in enumerate(rgb_stdet_config['data']['train']
                                         ['custom_classes'])
            }
    except KeyError:
        pass

    # 对整个视频事件进行预测
    action_result = None
    if args.use_skeleton_recog:
        print('Use skeleton-based recognition')
        action_result = skeleton_based_action_recognition(
            args, pose_results, num_frame, h, w)
    else:
        print('Use rgb-based recognition')
        action_result = rgb_based_action_recognition(args)

    # 对每个检测到的人进行原子级别的运动检测
    stdet_preds = None
    if args.use_skeleton_stdet:
        print('Use skeleton-based SpatioTemporal Action Detection')
        clip_len, frame_interval = 8, 1
        timestamps, stdet_preds = skeleton_based_stdet(args, stdet_label_map,
                                                       human_detections,
                                                       pose_results, num_frame,
                                                       clip_len,
                                                       frame_interval, h, w)
        for i in range(len(human_detections)):
            # 将框映射回原来的大小
            det = human_detections[i]
            det[:, 0:4:2] *= w_ratio
            det[:, 1:4:2] *= h_ratio
            human_detections[i] = torch.from_numpy(det[:, :4]).to(args.device)

    else:
        print('Use rgb-based SpatioTemporal Action Detection')
        for i in range(len(human_detections)):
            det = human_detections[i]
            det[:, 0:4:2] *= w_ratio
            det[:, 1:4:2] *= h_ratio
            human_detections[i] = torch.from_numpy(det[:, :4]).to(args.device)
        timestamps, stdet_preds = rgb_based_stdet(args, frames,
                                                  stdet_label_map,
                                                  human_detections, w, h,
                                                  new_w, new_h, w_ratio,
                                                  h_ratio)

    # 将检测结果和原子级的预测结果打包
    stdet_results = []
    for timestamp, prediction in zip(timestamps, stdet_preds):
        human_detection = human_detections[timestamp - 1]
        # stdet_results的大小随行为识别检测结果定，原原少于检测框
        stdet_results.append(
            pack_result(human_detection, prediction, new_h, new_w))

    def dense_timestamps(timestamps, n):
        """Make it nx frames.为了恢复原视频的帧数"""
        old_frame_interval = (timestamps[1] - timestamps[0])
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
        return new_frame_inds.astype(np.int64)

    dense_n = int(args.predict_stepsize / args.output_stepsize)
    output_timestamps = dense_timestamps(timestamps, dense_n)
    # print(output_timestamps)
    # print(np.array(output_timestamps).shape)
    frames = [
        cv2.imread(frame_paths[timestamp - 1])
        for timestamp in output_timestamps
    ]

    # 绘制
    print('Performing visualization')
    # 为了调用绘制骨骼的接口初始化一个模型
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 args.device)

    if args.use_skeleton_recog or args.use_skeleton_stdet:
        pose_results = [
            pose_results[timestamp - 1] for timestamp in output_timestamps
        ]

    vis_frames = visualize(frames, stdet_results, pose_results, action_result,
                           pose_model, max_num=1)
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
                                fps=args.output_fps)
    vid.write_videofile(args.out_filename)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)


if __name__ == '__main__':
    main()
