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
import copy as cp
from mmcv import DictAction

from mmaction.apis import inference_recognizer, init_recognizer
from demo.demo_video_structuralize_test import skeleton_based_action_recognition, skeleton_based_stdet

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

def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


PLATEBLUE = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
PLATEBLUE = PLATEBLUE.split('-')
PLATEBLUE = [hex2color(h) for h in PLATEBLUE]
PLATEGREEN = '004b23-006400-007200-008000-38b000-70e000'
PLATEGREEN = PLATEGREEN.split('-')
PLATEGREEN = [hex2color(h) for h in PLATEGREEN]

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

def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name

def visualize(frames,
              annotations,
              pose_results,
              action_result,
              pose_model,
              plate=PLATEBLUE,
              max_num=5):
    """Visualize frames with predicted annotations.

    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted spatio-temporal
            detection results.原子级预测tuple中为(bbox, )
        pose_results (list[list[tuple]): The pose results.
        action_result (str): The predicted action recognition results.时间级预测
        pose_model (nn.Module): The constructed pose model.
        plate (str): The plate used for visualization. Default: PLATEBLUE.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.

    Returns:
        list[np.ndarray]: Visualized frames.
    """
    # 色盘的大小必须大于max_num，否者弹出异常
    assert max_num + 1 <= len(plate)
    plate = [x[::-1] for x in plate]

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

                # 绘制检测框
                box = box.astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                if not pose_results:
                    cv2.rectangle(frame, st, ed, plate[0], 2)

                text = ': '.join([label, "%.2f%%"%(score*100)])
                print(text)
                location = (0 + st[0], 18 + 1 * 18 + st[1])
                textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                            THICKNESS)[0]
                textwidth = textsize[0]
                plate_width = 16
                diag0 = (location[0] + textwidth, location[1] - plate_width)
                diag1 = (location[0], location[1] + 2)
                cv2.rectangle(frame, diag0, diag1, plate[1], -1)
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

    return frames_

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

def skeleton_based_spatiotemporal_reconition(args, model, label_map, human_detections, pose_results,
                         num_frame, h, w, clip_len=1, frame_interval=1):
    # h与w我们不需要
    """
    args: 参数
    label_map: list，标签映射
    human_detections: 人体检测数据
    pose_result: 人体骨骼数据
    num_frame: 处理帧的数量
    clib_len: 默认为1
    frame_interval: 默认为1

    Returns:
    timestamps: 预测的视频帧index
    stdet_preds: list[list[list[tuple]]
    """
    # 时序骨骼长度为8
    windowsize = 8
    timestamps = np.arange(0, num_frame)

    # 加载模型
    skeleton_stdet_model = model

    # 将检测框和预测标签匹配起来
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

        # 建立人体骨骼的时序长度
        # start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        start_frame = timestamp - windowsize//2
        frame_inds = start_frame + np.arange(0, windowsize, frame_interval)
        frame_inds =[i for i in list(frame_inds) if i>0 and i< num_frame]
        pose_num_frame = len(frame_inds)

        # 取出该帧的人体骨骼
        pose_result = [pose_results[ind] for ind in frame_inds]

        # 对该帧中所有的人体骨骼进行检测
        skeleton_prediction = []
        for i in range(proposal.shape[0]):

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
            # result = {label_map[results[0][0]], results[0][1]}
            # st_action_label = label_map[results[0][0]]
            # print(st_action_label)
            pred = (area, label_map[results[0][0]], results[0][1])
            skeleton_prediction.append(pred)

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
    timestamps, stdet_preds = skeleton_based_spatiotemporal_reconition(args, model, label_map, det_results, pose_results, num_frame, h, w)
    
    # 逐一可视化每一帧中的骨架
    # 生成这个是为了更加方便画骨骼图
    print("Performing visualization")
    frames = [
        cv2.imread(frame_paths[timestamp])
        for timestamp in timestamps
        ]
    # pose_result也应该对齐
    # 不过这里我们使用的是一帧一帧的处理，所以这里可以忽视

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 args.device)
    print("\ndet",det_results[0][0])
    print("\npreds",stdet_preds[0])
    print("\npose",pose_results[0])
    print(action_label)
    vis_frames = visualize(frames, stdet_preds, pose_results, 
                           action_label, pose_model)

    # 写入视频
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    vid.write_videofile(args.out_filename, remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)


if __name__ == '__main__':
    main()
