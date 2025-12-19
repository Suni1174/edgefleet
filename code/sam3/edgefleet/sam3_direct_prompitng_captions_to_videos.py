"""
SAM3 Video Processing with configurable modes and batch processing.

env: /workspace/.common_cache/common_share_for_users/vlm_vidsitu/envs/sam3/bin/python

Issue: https://github.com/AKR-END/vlm_vidsitu/issues/13#issuecomment-3633829844

How to Run: CUDA_VISIBLE_DEVICES=2 python sam3_direct_prompitng_captions_to_videos.py
"""

import argparse
import os
import json
import logging
import traceback
from typing import Tuple
from anyio import Path
import torch
from sam3.model_builder import build_sam3_video_predictor
from utils import (
	load_config,
	setup_logging,
	get_video_events_file_path,
)
from process_video import process_single_video

def process_batch_videos(predictor, cfg, logger) -> Tuple[int, int]:
	"""Process multiple videos from JSON annotations."""
	successful_count = 0
	total_count = 0
	
	try:
		# Load annotations JSON
		logger.info(f"Loading annotations from: {cfg.batch_mode.annotations_json}")
		with open(cfg.batch_mode.annotations_json, 'r') as f:
			annotations = json.load(f)
		
		logger.info(f"Found {len(annotations)} videos in annotations")
		
		# Filter videos based on configuration
		all_video_ids = list(annotations.keys())
		video_ids = []
		
		# Apply video selection based on config
		if cfg.batch_mode.selected_video_ids:
			# Option 3: Select specific video IDs by name
			video_ids = [vid for vid in cfg.batch_mode.selected_video_ids if vid in all_video_ids]
			missing_ids = set(cfg.batch_mode.selected_video_ids) - set(all_video_ids)
			if missing_ids:
				logger.warning(f"Selected video IDs not found in annotations: {missing_ids}")
			logger.info(f"Selected {len(video_ids)} specific video IDs")
			
		elif cfg.batch_mode.selected_range:
			# Option 2: Select a range of indices
			start_idx, end_idx = cfg.batch_mode.selected_range
			start_idx = max(0, start_idx)
			end_idx = min(len(all_video_ids), end_idx)
			video_ids = all_video_ids[start_idx:end_idx]
			logger.info(f"Selected video range [{start_idx}:{end_idx}] = {len(video_ids)} videos")
			
		elif cfg.batch_mode.selected_indices:
			# Option 1: Select specific indices
			valid_indices = [i for i in cfg.batch_mode.selected_indices if 0 <= i < len(all_video_ids)]
			invalid_indices = [i for i in cfg.batch_mode.selected_indices if i < 0 or i >= len(all_video_ids)]
			if invalid_indices:
				logger.warning(f"Invalid indices (out of range): {invalid_indices}")
			video_ids = [all_video_ids[i] for i in valid_indices]
			logger.info(f"Selected {len(video_ids)} videos by indices: {valid_indices}")
			
		else:
			# Default: use all videos
			video_ids = all_video_ids
			logger.info(f"Processing all {len(video_ids)} videos")
		
		# Apply max_videos limit if specified
		if cfg.batch_mode.max_videos and len(video_ids) > cfg.batch_mode.max_videos:
			video_ids = video_ids[:cfg.batch_mode.max_videos]
			logger.info(f"Limited to {len(video_ids)} videos due to max_videos setting")
		
		for video_id in video_ids:
			total_count += 1
			logger.info(f"Processing video {total_count}/{len(video_ids)}: {video_id}")
			
			try:
				video_data = annotations[video_id]
				
				# Check if video has unique_args_for_grounding_in_vidsitu_order
				if 'unique_args_for_grounding_in_vidsitu_order' not in video_data:
					if cfg.batch_mode.skip_videos_without_args:
						logger.warning(f"Skipping {video_id}: no unique_args_for_grounding_in_vidsitu_order")
						continue
					else:
						logger.warning(f"No unique_args_for_grounding_in_vidsitu_order for {video_id}, using empty list")
						prompts = []
				else:
					prompts = video_data['unique_args_for_grounding_in_vidsitu_order']
					
				if not prompts:
					logger.warning(f"Skipping {video_id}: empty prompts list")
					continue
				
				# Construct video path
				video_path = os.path.join(cfg.batch_mode.base_mp4_folder, f"{video_id}.mp4")
				
				if not os.path.exists(video_path):
					logger.error(f"Video file not found: {video_path}")
					if not cfg.error_handling.continue_on_error:
						break
					continue
				
				# Create output directory for this video
				if cfg.output.create_video_subdirs:
					video_output_dir = os.path.join(cfg.output.base_dir, video_id)
				else:
					video_output_dir = cfg.output.base_dir
				
				os.makedirs(video_output_dir, exist_ok=True)
				
				# Get video events file path if shot processing is enabled
				video_events_file = None
				if cfg.get('shot_processing', {}).get('enabled', False):
					video_events_file = get_video_events_file_path(
						video_path, 
						cfg.shot_processing.get('video_events_base_dir')
					)
				
				# Process the video
				success = process_single_video(
					predictor=predictor,
					video_path=video_path,
					prompts=prompts,
					num_sample_frames=cfg.batch_mode.num_sample_frames,
					output_dir=video_output_dir,
					cfg=cfg,
					logger=logger,
					video_events_file=video_events_file
				)
				
				if success:
					successful_count += 1
				else:
					logger.error(f"Failed to process {video_id}")
					if not cfg.error_handling.continue_on_error:
						break
						
			except Exception as e:
				logger.error(f"Error processing {video_id}: {str(e)}")
				if logger.level <= logging.DEBUG:
					logger.error(traceback.format_exc())
				if not cfg.error_handling.continue_on_error:
					break
		
		logger.info(f"Batch processing completed: {successful_count}/{total_count} videos processed successfully")
		return successful_count, total_count
		
	except Exception as e:
		logger.error(f"Error in batch processing: {str(e)}")
		if logger.level <= logging.DEBUG:
			logger.error(traceback.format_exc())
		return successful_count, total_count

def main():
	"""Main function with configuration-based processing."""
	parser = argparse.ArgumentParser(description="SAM3 Video Processing")
	parser.add_argument(
		"--config", 
		type=str, 
		default="/home/suneetha/workspace/tmp/place/data_creation/sam3/edgefleet/config.yaml", 
		help="Path to configuration YAML file"
	)
	args = parser.parse_args()
	
	try:
		# Load configuration
		cfg = load_config(args.config)
		
		# Setup logging
		logger = setup_logging(cfg)
		logger.info("Starting SAM3 video processing")
		logger.info(f"Processing mode: {cfg.mode}")
		
		# Setup output directory
		os.makedirs(cfg.output.base_dir, exist_ok=True)
		
		# Setup GPUs
		if cfg.gpu.use_all_gpus:
			gpus_to_use = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
		elif cfg.gpu.device_ids:
			gpus_to_use = cfg.gpu.device_ids
		else:
			gpus_to_use = [0] if torch.cuda.is_available() else []
		
		logger.info(f"Using GPUs: {gpus_to_use}")
		
		# Build SAM3 predictor
		logger.info("Building SAM3 video predictor...")
		predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
		
		try:
			if cfg.mode == "single":
				base_dir = cfg.shot_processing.get('video_events_base_dir') if cfg.get('shot_processing', {}).get('enabled', False) else None
				video_file_stem = Path(cfg.single_mode.video_path).stem
				video_events_file = os.path.join(base_dir, video_file_stem, video_file_stem, f"{video_file_stem}.videvents") if base_dir else None
				os.makedirs(os.path.join(cfg.output.base_dir, video_file_stem), exist_ok=True)
				# Single video mode
				logger.info("Running in single video mode")
				success = process_single_video(
					predictor=predictor,
					video_path=cfg.single_mode.video_path,
					prompts=cfg.single_mode.prompts,
					num_sample_frames=cfg.single_mode.num_sample_frames,
					output_dir=os.path.join(cfg.output.base_dir, video_file_stem),
					cfg=cfg,
					logger=logger,
					video_events_file=video_events_file
				)
				
				if success:
					logger.info("Single video processing completed successfully")
				else:
					logger.error("Single video processing failed")
					
			elif cfg.mode == "batch":
				# Batch processing mode
				logger.info("Running in batch processing mode")
				successful, total = process_batch_videos(predictor, cfg, logger)
				logger.info(f"Batch processing results: {successful}/{total} videos processed successfully")
				
			else:
				logger.error(f"Unknown processing mode: {cfg.mode}")
				return 1
				
		finally:
			# Always shutdown predictor
			logger.info("Shutting down predictor...")
			predictor.shutdown()
		
		logger.info("Processing completed")
		return 0
		
	except Exception as e:
		if 'logger' in locals():
			logger.error(f"Fatal error: {str(e)}")
			if logger.level <= logging.DEBUG:
				logger.error(traceback.format_exc())
		else:
			print(f"Fatal error: {str(e)}")
			traceback.print_exc()
		return 1

if __name__ == "__main__":
	main()