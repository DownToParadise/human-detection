# 增加原子级动作检测
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil

import cv2
import mmcv
import numpy as np
import torch
from mmcv import DictAction

from mmaction.apis import inference_recognizer, init_recognizer
from demo_video_structuralize import skeleton_based_action_recognition, skeleton_based_stdet

from mmaction.datasets.pipelines import Compose
from mmcv.runner import load_checkpoint
from mmaction.models import build_model

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this demo! ')

try:
    from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                             vis_pose_result)
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model`, '
                      '`init_pose_model`, and `vis_pose_result` form '
                      '`mmpose.apis`. These apis are required in this demo! ')

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    # 参数配置
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('video', default='demo/demo.mp4', help='video file/url')
    parser.add_argument('out_filename', default='demo/demo_out.mp4', help='output filename')
    parser.add_argument(
        '--config',
        default=('configs/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu120_xsub_keypoint.py'),
        help='skeleton model config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu120_xsub_keypoint/'
                 'slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth'),
        help='skeleton model checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
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
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label-map',
        default='tools/data/skeleton/label_map_ntu120.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
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


def frame_extraction(video_path, short_side):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    # cv2读取帧
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    # 下面表示逐帧处理
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        # 对抽取的帧编号
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        # 写入抽取帧到指定目录
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
        # 进度条更新
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
        # 得到人体骨骼信息
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret

def cal_iou(box1, box2):
    """计算两个框的交并比"""
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

def expand_bbox(person_bbox, h, w):
    """为了后期使框更加平滑，先占个坑位"""
    return person_bbox

def skeleton_based_spatiotemporal_reconition(args, label_map, human_detections, pose_results,
                         num_frame, h, w, clip_len=1, frame_interval=1):
    #h与w我们不需要
    """
    args: 参数
    label_map: list，标签映射
    human_detections: 人体检测数据
    pose_result: 人体骨骼数据
    num_frame: 处理帧的数量
    clib_len: 默认为1
    frame_interval: 默认为1

    Returns:
    每一帧中识别道德骨架的预测
    """
    # 时序骨骼长度为8
    windowsize = 8
    timestamps = np.arange(0, num_frame)

    # 加载模型
    skeleton_config = mmcv.Config.fromfile(args.config)
    num_class = len(label_map)
    skeleton_pipeline = Compose(skeleton_config.test_pipeline)

    # 加载模型
    skeleton_stdet_model = build_model(skeleton_config.model)
    load_checkpoint(
        skeleton_stdet_model,
        args.checkpoint,
        map_location='cpu')
    skeleton_stdet_model.to(args.device)
    skeleton_stdet_model.eval()

    skeleton_predictions = []

    print('Performing SpatioTemporal Action Detection for each clip')
    # 按照选定的帧数进行预测
    prog_bar = mmcv.ProgressBar(len(timestamps))
    for timestamp in timestamps:
        # proposal与pose_result的帧数要对齐
        # 取出该帧的人体检测
        proposal = human_detections[timestamp]
        if proposal.shape[0] == 0:  # no people detected
            skeleton_predictions.append(None)
            continue
        
        # 取出该帧的人体骨骼
        print(timestamp)
        pose_result = pose_results[timestamp]

        # 建立人体骨骼的时序长度
        # start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        start_frame = timestamp - windowsize//2
        frame_inds = start_frame + np.arange(0, windowsize, frame_interval)
        frame_inds =[i for i in list(frame_inds) if i>0 and i< num_frame]
        pose_num_frame = len(frame_inds)

        # 对该帧中所有的人体骨骼进行检测
        skeleton_prediction = []
        for i in range(proposal.shape[0]):
            # 这一步是为了后期对齐数据
            skeleton_prediction.append([])

            # 为每个骨骼建立数据
            fake_anno = dict(
                frame_dict='',
                label=-1,
                img_shape=(h, w),
                origin_shape=(h, w),
                start_index=0,
                modality='Pose',
                total_frames=pose_num_frame)
            num_person = 1

            num_keypoint = 17
            keypoint = np.zeros(
                (num_person, pose_num_frame, num_keypoint, 2))  # M T V 2
            keypoint_score = np.zeros(
                (num_person, pose_num_frame, num_keypoint))  # M T V
            
            # 得到人体检测的具体位置
            person_bbox = proposal[i][:4]
            area = expand_bbox(person_bbox, h, w)
            # j为某一帧，poses为该帧中的所有人体骨骼
            # 这里因为pose_result中只有一帧的骨骼信息，所以只用套一层循环
            for j, poses in enumerate(pose_result):  
                # 将人体骨骼和检测框进行匹配
                max_iou = float('-inf')
                index = -1
                if len(poses) == 0:
                    continue
                for k, per_pose in enumerate(poses):
                    # 为什么这个地方perpose不是字典而是字符串
                    iou = cal_iou(per_pose['bbox'][:4], area)
                    if max_iou < iou:
                        index = k
                        max_iou = iou
                # 尽管这里又很多帧，但是由于area只有一个检测框，所以只能匹配一个骨骼
                keypoint[0, j] = poses[index]['keypoints'][:, :2]
                keypoint_score[0, j] = poses[index]['keypoints'][:, 2]

            fake_anno['keypoint'] = keypoint
            fake_anno['keypoint_score'] = keypoint_score

            # pipeline处理
            # return返回字典，前五个精度最高的标签和分数
            results = inference_recognizer(skeleton_stdet_model, fake_anno)
            skeleton_prediction.append(results[0])
            st_action_label = label_map[results[0][0]]
            print(st_action_label)

    skeleton_predictions.append(skeleton_prediction)
    prog_bar.update()
    return timestamps, skeleton_predictions



def main():
    args = parse_args()

    # 这里是对一段视频进行检测，而不是进行一个实时的检测
    frame_paths, original_frames = frame_extraction(args.video,
                                                    args.short_side)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # Get clip_len, frame_interval and calculate center index of each clip
    config = mmcv.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)
    for component in config.data.test.pipeline:
        if component['type'] == 'PoseNormalize':
            component['mean'] = (w // 2, h // 2, .5)
            component['max_value'] = (w, h, 1.)

    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()]

    # 得到人体检测信息
    det_results = detection_inference(args, frame_paths)
    torch.cuda.empty_cache()

    # 得到骨架信息
    pose_results = pose_inference(args, frame_paths, det_results)
    torch.cuda.empty_cache()

    # 构造这个字典方便后期函数传参
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    
    # 得到该视频中出现的最多的人数
    num_person = max([len(x) for x in pose_results])

    # 根据人数初始化数据结构
    num_keypoint = 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                              dtype=np.float16)
    
    # 得到i帧中j人的关节点信息，和检测器检测其是否出现的信息
    for i, poses in enumerate(pose_results):
        for j, pose in enumerate(poses):
            pose = pose['keypoints']
            # 得到关节点信息
            keypoint[j, i] = pose[:, :2]
            # 得到人体信息置信度
            keypoint_score[j, i] = pose[:, 2]
    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score

    # 如果我不把所有人的骨架信息送进去，只送单独一人的信息，是不是就可以识别原子级的动作
    # 这里要打印以下这个results长啥样
    # 加载预训练模型
    model = init_recognizer(config, args.checkpoint, args.device)
    # 得到群体结果最大的前五个
    results = inference_recognizer(model, fake_anno)
    
    # 得到群体行为的结果
    # 问题是现在我们要得到单个运动人的结果
    print(results)
    action_label = label_map[results[0][0]]

    # 原子级动作检测
    # 这几个数据的个数都是一致的先尝试逐帧检测
    # print("\nlabel_map", np.array(label_map).shape, "det_results", np.array(det_results).shape, "pose_results", np.array(pose_results).shape, "num_frame", num_frame, sep="\t")
    skeleton_based_spatiotemporal_reconition(args, label_map, det_results, pose_results, num_frame, h, w)
    # 生成这个是为了更加方便画骨骼图
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 args.device)
    
    # 逐一可视化每一帧中的骨架
    vis_frames = [
        vis_pose_result(pose_model, frame_paths[i], pose_results[i])
        for i in range(num_frame)
    ]
    for frame in vis_frames:
        cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)

    # 写入视频
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    vid.write_videofile(args.out_filename, remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)


if __name__ == '__main__':
    main()
