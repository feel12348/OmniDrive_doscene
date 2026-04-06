import argparse
import csv
import importlib
import os
import os.path as osp
import sys
from collections import defaultdict

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description='doScenes full-scene sequential evaluation (single GPU)')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--doscenes-csv',
        default='data/annotated_doscenes.csv',
        help='CSV with doScenes instructions')
    parser.add_argument(
        '--save-dir',
        default=None,
        help='override model.save_path (directory for generated text)')
    parser.add_argument(
        '--index-json',
        default=None,
        help='optional json path to dump output metadata index')
    parser.add_argument(
        '--max-scenes',
        type=int,
        default=0,
        help='debug option: only process first N scenes (0 means all)')
    parser.add_argument(
        '--max-frames-per-scene',
        type=int,
        default=0,
        help='debug option: only process first N frames per scene (0 means all)')
    parser.add_argument(
        '--frame-stride',
        type=int,
        default=1,
        help='evaluate one frame every N frames in each scene (>=1)')
    parser.add_argument(
        '--fps',
        type=float,
        default=2.0,
        help='frame rate used to convert history-seconds to warm-up frames')
    parser.add_argument(
        '--history-seconds',
        type=float,
        default=2.0,
        help='warm-up seconds before applying instruction')
    parser.add_argument(
        '--history-frames',
        type=int,
        default=0,
        help='if >0, overrides history-seconds')
    parser.add_argument(
        '--drop-tail-frames',
        type=int,
        default=0,
        help='drop last N frames of each scene from sequential evaluation')
    parser.add_argument(
        '--record-warmup-outputs',
        action='store_true',
        help='if set, save outputs of warm-up frames; default skip saving them')
    parser.add_argument(
        '--no-instruction',
        action='store_true',
        help='if set, ignore doScenes language instruction and use empty text')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=mmcv.DictAction,
        help='override some settings in config')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher for multi-gpu evaluation')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def load_instruction_map(csv_path):
    scene_to_insts = defaultdict(list)
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_number = row.get('Scene Number')
            instruction = (row.get('Instruction') or '').strip()
            if not scene_number or not instruction:
                continue
            try:
                scene_name = f"scene-{int(float(scene_number)):04d}"
            except ValueError:
                continue
            scene_to_insts[scene_name].append(instruction)
    return scene_to_insts


def maybe_import_plugin(cfg, config_path):
    if not hasattr(cfg, 'plugin') or not cfg.plugin:
        return
    if hasattr(cfg, 'plugin_dir'):
        module_dir = os.path.dirname(cfg.plugin_dir)
    else:
        module_dir = os.path.dirname(config_path)
    module_parts = module_dir.split('/')
    module_path = module_parts[0]
    for part in module_parts[1:]:
        if part:
            module_path = module_path + '.' + part
    importlib.import_module(module_path)


def prepare_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    maybe_import_plugin(cfg, args.config)
    cfg.model.pretrained = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    else:
        raise TypeError('Only dict cfg.data.test is supported in this script.')
    return cfg


def build_model_and_dataset(cfg, checkpoint_path, save_dir=None, device_id=0):
    dataset = build_dataset(cfg.data.test)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    if save_dir is not None:
        model.save_path = save_dir
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(device_id), device_ids=[device_id])
    model.eval()
    return model, dataset


def setup_distributed(args):
    if args.launcher == 'none':
        torch.cuda.set_device(0)
        return 0, 1, 0

    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank


def is_main_process(rank):
    return rank == 0


def gather_records(records, rank, world_size):
    if world_size == 1:
        return records
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, records)
    if rank == 0:
        merged = []
        for part in gathered:
            merged.extend(part)
        return merged
    return None


def gather_int(value, rank, world_size):
    if world_size == 1:
        return value
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, int(value))
    if rank == 0:
        return int(sum(gathered))
    return None


def reset_temporal_memory(model):
    module = model.module if hasattr(model, 'module') else model
    if hasattr(module, 'pts_bbox_head') and module.with_pts_bbox:
        module.pts_bbox_head.reset_memory()
    if hasattr(module, 'map_head') and module.with_map_head:
        module.map_head.reset_memory()
    module.test_flag = True


def inject_runtime_meta(batch, skip_save=False, output_name=None):
    if 'img_metas' not in batch:
        return

    def _inject(obj):
        if hasattr(obj, 'data'):
            _inject(obj.data)
            return
        if isinstance(obj, dict):
            if 'sample_idx' in obj:
                obj['skip_save'] = bool(skip_save)
                if output_name is not None:
                    obj['output_name'] = output_name
            for value in obj.values():
                _inject(value)
            return
        if isinstance(obj, list):
            for item in obj:
                _inject(item)
            return
        if isinstance(obj, tuple):
            for item in obj:
                _inject(item)

    _inject(batch['img_metas'])


def run_single_sample(model, dataset, index, instruction, output_name=None, skip_save=False):
    scene_name = dataset.data_infos[index].get('scene_name', '')
    if hasattr(dataset, 'doscenes_map'):
        dataset.doscenes_map[scene_name] = instruction

    sample = dataset[index]
    batch = collate([sample], samples_per_gpu=1)
    inject_runtime_meta(batch, skip_save=skip_save, output_name=output_name)
    with torch.no_grad():
        _ = model(return_loss=False, rescale=True, **batch)


def build_scene_indices(dataset):
    scene_to_indices = defaultdict(list)
    for idx, info in enumerate(dataset.data_infos):
        scene_to_indices[info.get('scene_name', '')].append(idx)
    for scene_name, indices in scene_to_indices.items():
        indices.sort(key=lambda i: dataset.data_infos[i].get('frame_idx', i))
        scene_to_indices[scene_name] = indices
    return scene_to_indices


def get_history_frames(args):
    if args.history_frames > 0:
        return args.history_frames
    return max(int(round(args.history_seconds * args.fps)), 0)


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed(args)
    cfg = prepare_cfg(args)
    model, dataset = build_model_and_dataset(
        cfg, args.checkpoint, save_dir=args.save_dir, device_id=local_rank)

    if args.save_dir is None:
        save_dir = model.module.save_path
    else:
        save_dir = args.save_dir
    mmcv.mkdir_or_exist(save_dir)

    scene_to_insts = load_instruction_map(args.doscenes_csv)
    scene_to_indices = build_scene_indices(dataset)
    history_frames = get_history_frames(args)

    selected_scenes = [s for s in scene_to_indices.keys() if s in scene_to_insts]
    selected_scenes.sort()
    if args.max_scenes > 0:
        selected_scenes = selected_scenes[:args.max_scenes]

    metadata_index = []
    local_total_outputs = 0
    if is_main_process(rank):
        print(
            f'[FullSceneEval] scenes={len(selected_scenes)}, '
            f'history_frames={history_frames}, frame_stride={args.frame_stride}, '
            f'world_size={world_size}'
        )

    instruction_task_id = 0
    for scene_pos, scene_name in enumerate(selected_scenes):
        frame_indices = scene_to_indices[scene_name]
        if args.max_frames_per_scene > 0:
            frame_indices = frame_indices[:args.max_frames_per_scene]
        if args.frame_stride > 1:
            frame_indices = frame_indices[::args.frame_stride]
        instructions = scene_to_insts[scene_name]

        if is_main_process(rank):
            print(
                f'[FullSceneEval] scene {scene_pos + 1}/{len(selected_scenes)} '
                f'{scene_name}: frames={len(frame_indices)}, instructions={len(instructions)}'
            )

        for inst_id, instruction_text in enumerate(instructions):
            process_this_instruction = (instruction_task_id % world_size == rank)
            instruction_task_id += 1
            if not process_this_instruction:
                continue

            reset_temporal_memory(model)

            # Warm-up phase: advance temporal memory without instruction.
            warm_end = min(history_frames, len(frame_indices))
            for warm_pos in range(warm_end):
                warm_idx = frame_indices[warm_pos]
                info = dataset.data_infos[warm_idx]
                sample_token = info['token']
                frame_idx = int(info.get('frame_idx', warm_pos))
                output_name = (
                    f'{sample_token}__{scene_name}__inst{inst_id:03d}'
                    f'__warmup__frame{frame_idx:04d}.json'
                )
                run_single_sample(
                    model=model,
                    dataset=dataset,
                    index=warm_idx,
                    instruction='',
                    output_name=output_name if args.record_warmup_outputs else None,
                    skip_save=(not args.record_warmup_outputs),
                )

            # Sequential full-scene phase: apply instruction from warm_end to
            # end_pos (optionally dropping tail frames for fair comparison).
            end_pos = len(frame_indices) - max(args.drop_tail_frames, 0)
            if end_pos <= warm_end:
                continue
            for pos in range(warm_end, end_pos):
                idx = frame_indices[pos]
                info = dataset.data_infos[idx]
                sample_token = info['token']
                frame_idx = int(info.get('frame_idx', pos))
                instruction = '' if args.no_instruction else instruction_text
                output_name = (
                    f'{sample_token}__{scene_name}__inst{inst_id:03d}'
                    f'__seq__frame{frame_idx:04d}.json'
                )
                run_single_sample(
                    model=model,
                    dataset=dataset,
                    index=idx,
                    instruction=instruction,
                    output_name=output_name,
                    skip_save=False,
                )

                metadata_index.append(
                    dict(
                        output_name=output_name,
                        sample_token=sample_token,
                        scene_name=scene_name,
                        frame_idx=frame_idx,
                        instruction_id=inst_id,
                        instruction=instruction_text,
                        applied_instruction=(instruction != ''),
                        phase='sequential',
                    )
                )
                local_total_outputs += 1
                if local_total_outputs % 100 == 0:
                    print(f'[FullSceneEval][rank{rank}] completed outputs: {local_total_outputs}')

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
    all_records = gather_records(metadata_index, rank, world_size)
    total_outputs = gather_int(local_total_outputs, rank, world_size)

    if is_main_process(rank):
        print(f'[FullSceneEval] done. total outputs={total_outputs}, save_dir={save_dir}')
        if args.index_json:
            mmcv.dump(all_records, args.index_json)
            print(f'[FullSceneEval] index saved to {args.index_json}')


if __name__ == '__main__':
    main()
