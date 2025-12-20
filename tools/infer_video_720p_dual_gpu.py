# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
"""
Dual GPU parallel inference script for video generation.
This script reads prompts from BLIP-caption txt files and generates videos
using two A800 GPUs in parallel.

Video naming convention: video{file_idx}-{line_idx}.mp4
- file_idx: 1 for 1st_10min.txt, 2 for 2nd_10min.txt, etc.
- line_idx: line number in the txt file (1-indexed)
"""

import sys
import os
import os.path as osp
import time
import glob
import re
from multiprocessing import Process, Queue
import numpy as np
import torch
import argparse

sys.path.append(osp.dirname(osp.dirname(__file__)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_file_index(filename):
    """Extract file index from filename like '1st_10min.txt' -> 1"""
    match = re.match(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def load_prompts_from_folder(folder_path):
    """
    Load all prompts from txt files in the folder.
    Returns a list of tuples: (file_idx, line_idx, prompt)
    """
    tasks = []
    txt_files = sorted(glob.glob(osp.join(folder_path, "*.txt")))
    
    for txt_file in txt_files:
        filename = osp.basename(txt_file)
        file_idx = get_file_index(filename)
        
        if file_idx is None:
            print(f"Warning: Cannot extract index from {filename}, skipping...")
            continue
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_idx, line in enumerate(lines, start=1):
            prompt = line.strip()
            if prompt:  # Skip empty lines
                tasks.append((file_idx, line_idx, prompt))
    
    return tasks


def worker_process(gpu_id, task_queue, result_queue, args_dict, save_dir):
    """
    Worker process that runs on a specific GPU.
    """
    # Set GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Import after setting CUDA_VISIBLE_DEVICES
    import torch
    import cv2
    from PIL import Image
    
    from tools.run_infinity import (
        load_tokenizer, load_transformer, load_visual_tokenizer, 
        gen_one_example, save_video, transform
    )
    from infinity.models.self_correction import SelfCorrection
    from infinity.schedules.dynamic_resolution import (
        get_dynamic_resolution_meta, 
        get_first_full_spatial_size_scale_index
    )
    from infinity.schedules import get_encode_decode_func
    from infinity.utils.arg_util import Args
    
    print(f"[GPU {gpu_id}] Initializing...")
    
    # Create args object
    args = Args()
    for key, value in args_dict.items():
        setattr(args, key, value)
    
    # Load models
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    vae = vae.float().to('cuda')
    infinity = load_transformer(vae, args)
    self_correction = SelfCorrection(vae, args)
    
    video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(
        args.dynamic_scale_schedule
    )
    
    print(f"[GPU {gpu_id}] Models loaded successfully!")
    
    # Inference parameters
    mapped_duration = 5
    num_frames = 81
    seed = 41
    
    # Get dynamic resolution meta
    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(
        args.dynamic_scale_schedule, args.video_frames
    )
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 0.571))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames-1)//4+1]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)
    tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(scale_schedule) - args.tower_split_index)
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
    
    # Create output directory
    gen_video_dir = osp.join(save_dir, 'gen_videos')
    os.makedirs(gen_video_dir, exist_ok=True)
    
    # Process tasks from queue
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill to stop worker
            print(f"[GPU {gpu_id}] Received stop signal, exiting...")
            break
        
        file_idx, line_idx, prompt = task
        video_name = f"video{file_idx}-{line_idx}.mp4"
        video_path = osp.join(gen_video_dir, video_name)
        
        # Skip if already exists
        if osp.exists(video_path):
            print(f"[GPU {gpu_id}] Skipping {video_name} (already exists)")
            result_queue.put((file_idx, line_idx, video_path, 0, "skipped"))
            continue
        
        print(f"[GPU {gpu_id}] Processing {video_name}: {prompt[:50]}...")
        
        try:
            # Prepare prompt
            full_prompt = prompt
            if args.append_duration2caption:
                full_prompt = f'<<<t={mapped_duration}s>>>' + full_prompt
            
            start_time = time.time()
            
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
                generated_image, _ = gen_one_example(
                    infinity,
                    vae,
                    text_tokenizer,
                    text_encoder,
                    full_prompt,
                    negative_prompt="",
                    g_seed=seed,
                    gt_leak=-1,
                    gt_ls_Bl=None,
                    cfg_list=args.cfg,
                    tau_list=tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[0],
                    vae_type=args.vae_type,
                    sampling_per_bits=1,
                    enable_positive_prompt=0,
                    low_vram_mode=True,
                    args=args,
                    get_visual_rope_embeds=get_visual_rope_embeds,
                    context_info=context_info,
                    noise_list=None,
                )
                
                if len(generated_image.shape) == 3:
                    generated_image = generated_image.unsqueeze(0)
            
            elapsed_time = time.time() - start_time
            
            # Save video
            save_video(generated_image.cpu().numpy(), fps=args.fps, save_filepath=video_path)
            
            print(f"[GPU {gpu_id}] Saved {video_name} in {elapsed_time:.2f}s")
            result_queue.put((file_idx, line_idx, video_path, elapsed_time, "success"))
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {video_name}: {str(e)}")
            result_queue.put((file_idx, line_idx, video_path, 0, f"error: {str(e)}"))
    
    print(f"[GPU {gpu_id}] Worker finished.")


def main():
    parser = argparse.ArgumentParser(description='Dual GPU parallel video inference')
    parser.add_argument('--caption_folder', type=str, 
                        default='./eeg_data/Video/BLIP-caption',
                        help='Path to folder containing caption txt files')
    parser.add_argument('--output_dir', type=str, 
                        default='./infinity_output_gt',
                        help='Output directory for generated videos')
    parser.add_argument('--checkpoints_dir', type=str, 
                        default='./checkpoints/',
                        help='Path to model checkpoints')
    parser.add_argument('--gpu_ids', type=str, 
                        default='0,1',
                        help='GPU IDs to use (comma-separated)')
    cli_args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in cli_args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)
    print(f"Using {num_gpus} GPUs: {gpu_ids}")
    
    # Load all tasks
    print(f"Loading prompts from {cli_args.caption_folder}...")
    tasks = load_prompts_from_folder(cli_args.caption_folder)
    print(f"Total tasks: {len(tasks)}")
    
    if len(tasks) == 0:
        print("No tasks found! Please check the caption folder.")
        return
    
    # Prepare args dict for workers
    checkpoints_dir = cli_args.checkpoints_dir
    args_dict = {
        'pn': '0.90M',
        'fps': 16,
        'video_frames': 81,
        'model_path': osp.join(checkpoints_dir, 'infinitystar_8b_720p_weights'),
        'checkpoint_type': 'torch_shard',
        'vae_path': osp.join(checkpoints_dir, 'infinitystar_videovae.pth'),
        'text_encoder_ckpt': osp.join(checkpoints_dir, 'text_encoder/flan-t5-xl-official/'),
        'model_type': 'infinity_qwen8b',
        'text_channels': 2048,
        'dynamic_scale_schedule': 'infinity_elegant_clip20frames_v2',
        'bf16': 1,
        'use_apg': 1,
        'use_cfg': 0,
        'cfg': 34,
        'tau_image': 1,
        'tau_video': 0.4,
        'apg_norm_threshold': 0.05,
        'image_scale_repetition': '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]',
        'video_scale_repetition': '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]',
        'append_duration2caption': 1,
        'use_two_stage_lfq': 1,
        'detail_scale_min_tokens': 750,
        'semantic_scales': 12,
        'max_repeat_times': 10000,
        'enable_rewriter': 0,
    }
    
    # Create queues
    task_queue = Queue()
    result_queue = Queue()
    
    # Create output directory
    save_dir = cli_args.output_dir
    os.makedirs(osp.join(save_dir, 'gen_videos'), exist_ok=True)
    
    # Start worker processes
    workers = []
    for gpu_id in gpu_ids:
        p = Process(
            target=worker_process,
            args=(gpu_id, task_queue, result_queue, args_dict, save_dir)
        )
        p.start()
        workers.append(p)
    
    # Add tasks to queue
    for task in tasks:
        task_queue.put(task)
    
    # Add poison pills to stop workers
    for _ in workers:
        task_queue.put(None)
    
    # Collect results
    completed = 0
    success_count = 0
    error_count = 0
    skip_count = 0
    total_time = 0
    
    start_time = time.time()
    
    while completed < len(tasks):
        result = result_queue.get()
        file_idx, line_idx, video_path, elapsed_time, status = result
        completed += 1
        
        if status == "success":
            success_count += 1
            total_time += elapsed_time
        elif status == "skipped":
            skip_count += 1
        else:
            error_count += 1
        
        # Progress report
        if completed % 10 == 0 or completed == len(tasks):
            print(f"Progress: {completed}/{len(tasks)} "
                  f"(Success: {success_count}, Skipped: {skip_count}, Errors: {error_count})")
    
    # Wait for workers to finish
    for p in workers:
        p.join()
    
    total_elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"All tasks completed!")
    print(f"Total: {len(tasks)}, Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")
    print(f"Total time: {total_elapsed:.2f}s")
    if success_count > 0:
        print(f"Average time per video (excluding skipped): {total_time/success_count:.2f}s")
    print(f"Videos saved to: {osp.join(save_dir, 'gen_videos')}")


if __name__ == '__main__':
    main()

