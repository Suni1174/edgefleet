import os
import logging
import traceback
from altair import sample
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import List
import pickle
import json
from utils import (
	load_video_frames_for_vis,
	visualize_frame_with_masks,
	create_track_visualization_video,
	read_shot_boundaries,
	get_video_events_file_path,
	get_shot_segments,
	create_temp_shot_video,
	cleanup_temp_directory,
	load_molmo_points_for_video,
	get_video_properties
)

def propagate_in_video(predictor, session_id):
	"""Propagate predictions through the video using the predictor stream API.

	Returns: dict mapping frame_index -> outputs (as returned by predictor)
	"""
	outputs_per_frame = {}
	for response in predictor.handle_stream_request(
		request=dict(type="propagate_in_video", session_id=session_id)
	):
		outputs_per_frame[response["frame_index"]] = response["outputs"]
	return outputs_per_frame


def generate_per_frame_annotations(outputs_per_frame: dict, prompt_to_global_ids: dict, output_dir: str, logger):
	"""
	Generate per-frame annotation files containing frame index, x centroid, y centroid, and visibility flag.
	
	Args:
		outputs_per_frame: Dictionary of frame outputs containing masks and object IDs
		prompt_to_global_ids: Dictionary mapping prompts to global IDs
		output_dir: Directory to save annotation files
		logger: Logger instance
	"""
	try:
		# Create reverse mapping from global ID to prompt
		global_id_to_prompt = {}
		for prompt, global_ids in prompt_to_global_ids.items():
			for global_id in global_ids:
				global_id_to_prompt[global_id] = prompt
		
		# Get all unique object IDs across all frames
		all_obj_ids = set()
		for frame_outputs in outputs_per_frame.values():
			if isinstance(frame_outputs, dict) and 'out_obj_ids' in frame_outputs:
				if frame_outputs['out_obj_ids'] is not None:
					all_obj_ids.update(frame_outputs['out_obj_ids'])
		
		# Create annotation data for each object ID
		for obj_id in sorted(all_obj_ids):
			prompt_name = global_id_to_prompt.get(obj_id, f"object_{obj_id}")
			safe_prompt_name = prompt_name.replace(" ", "_").replace("/", "_")
			
			annotations = []
			
			# Process each frame
			for frame_idx in sorted(outputs_per_frame.keys()):
				frame_outputs = outputs_per_frame[frame_idx]
				
				# Initialize default values
				x_centroid = None
				y_centroid = None
				visibility = 0
				
				if isinstance(frame_outputs, dict):
					obj_ids = frame_outputs.get('out_obj_ids', [])
					masks = frame_outputs.get('out_binary_masks', [])
					
					# Check if this object appears in this frame
					if obj_ids is not None and obj_id in obj_ids:
						obj_index = list(obj_ids).index(obj_id)
						
						if masks is not None and obj_index < len(masks):
							mask = masks[obj_index]
							
							if mask is not None and mask.sum() > 0:
								# Compute centroid from mask
								coords = np.where(mask)
								if len(coords[0]) > 0:
									y_centroid = float(np.mean(coords[0]))
									x_centroid = float(np.mean(coords[1]))
									visibility = 1
				
				# Add annotation for this frame
				annotations.append({
					"frame_index": int(frame_idx),
					"x_centroid": x_centroid,
					"y_centroid": y_centroid,
					"visibility": visibility
				})
			
			# Save annotations for this object
			if annotations:
				annotation_file = os.path.join(output_dir, f"annotations_{safe_prompt_name}_obj{obj_id}.json")
				with open(annotation_file, 'w') as f:
					json.dump({
						"object_id": int(obj_id),
						"prompt": prompt_name,
						"total_frames": len(annotations),
						"annotations": annotations
					}, f, indent=2)
				
				logger.info(f"Saved per-frame annotations for '{prompt_name}' (obj_id: {obj_id}) to {annotation_file}")
		
		# Also create a combined annotation file with all objects
		if all_obj_ids:
			combined_annotations = []
			
			for frame_idx in sorted(outputs_per_frame.keys()):
				frame_data = {"frame_index": int(frame_idx), "objects": []}
				frame_outputs = outputs_per_frame[frame_idx]
				
				if isinstance(frame_outputs, dict):
					obj_ids = frame_outputs.get('out_obj_ids', [])
					masks = frame_outputs.get('out_binary_masks', [])
					
					if obj_ids is not None and masks is not None:
						for i, obj_id in enumerate(obj_ids):
							if i < len(masks) and masks[i] is not None:
								mask = masks[i]
								
								x_centroid = None
								y_centroid = None
								visibility = 0
								
								if mask.sum() > 0:
									coords = np.where(mask)
									if len(coords[0]) > 0:
										y_centroid = float(np.mean(coords[0]))
										x_centroid = float(np.mean(coords[1]))
										visibility = 1
								
								prompt_name = global_id_to_prompt.get(obj_id, f"object_{obj_id}")
								
								frame_data["objects"].append({
									"object_id": int(obj_id),
									"prompt": prompt_name,
									"x_centroid": x_centroid,
									"y_centroid": y_centroid,
									"visibility": visibility
								})
				
				combined_annotations.append(frame_data)
			
			# Save combined annotations
			combined_file = os.path.join(output_dir, "annotations_all_objects.json")
			with open(combined_file, 'w') as f:
				json.dump({
					"total_frames": len(combined_annotations),
					"total_objects": len(all_obj_ids),
					"prompt_to_global_ids": prompt_to_global_ids,
					"annotations": combined_annotations
				}, f, indent=2)
			
			logger.info(f"Saved combined per-frame annotations to {combined_file}")
			
	except Exception as e:
		logger.error(f"Failed to generate per-frame annotations: {str(e)}")
		if logger.level <= logging.DEBUG:
			logger.error(traceback.format_exc())


def normalize_object_ids(all_outputs_per_frame: dict, prompt_to_global_ids: dict, logger) -> dict:
	"""
	Normalize object IDs so each prompt has a single canonical global ID,
	then remap all IDs to continuous range.
	
	Args:
		all_outputs_per_frame: Dictionary of frame outputs containing object IDs
		prompt_to_global_ids: Dictionary mapping prompts to lists of global IDs
		logger: Logger instance for debugging
		
	Returns:
		Updated prompt_to_global_ids dictionary with normalized mappings
	"""
	logger.info("Normalizing object IDs to single ID per prompt...")
	
	# Step 1: Create mapping from all IDs to canonical ID for each prompt
	id_to_canonical_id = {}
	canonical_prompt_to_global_ids = {}
	
	for prompt, global_ids in prompt_to_global_ids.items():
		if global_ids:  # Only process prompts that have associated IDs
			canonical_id = global_ids[0]  # Use first ID as canonical
			canonical_prompt_to_global_ids[prompt] = [canonical_id]
			
			# Map all IDs for this prompt to the canonical ID
			for gid in global_ids:
				id_to_canonical_id[gid] = canonical_id
	
	# Step 2: Apply canonical ID mapping to all frame outputs
	for frame_idx, frame_outputs in all_outputs_per_frame.items():
		if 'out_obj_ids' in frame_outputs and frame_outputs['out_obj_ids']:
			# Update object IDs to canonical IDs
			updated_ids = []
			for obj_id in frame_outputs['out_obj_ids']:
				canonical_id = id_to_canonical_id.get(obj_id, obj_id)
				updated_ids.append(canonical_id)
			frame_outputs['out_obj_ids'] = updated_ids
	
	# Step 3: Create continuous ID remapping
	unique_canonical_ids = sorted(set(id_to_canonical_id.values()))
	continuous_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_canonical_ids)}
	
	logger.info(f"Remapping {len(unique_canonical_ids)} canonical IDs to continuous range 0-{len(unique_canonical_ids)-1}")
	
	# Step 4: Apply continuous remapping to frame outputs
	for frame_idx, frame_outputs in all_outputs_per_frame.items():
		if 'out_obj_ids' in frame_outputs and frame_outputs['out_obj_ids']:
			# Update object IDs to continuous range
			updated_ids = []
			for obj_id in frame_outputs['out_obj_ids']:
				continuous_id = continuous_id_mapping.get(obj_id, obj_id)
				updated_ids.append(continuous_id)
			frame_outputs['out_obj_ids'] = updated_ids
	
	# Step 5: Update prompt_to_global_ids mapping to use continuous IDs
	final_prompt_to_global_ids = {}
	for prompt, global_ids in canonical_prompt_to_global_ids.items():
		if global_ids:
			canonical_id = global_ids[0]
			continuous_id = continuous_id_mapping.get(canonical_id, canonical_id)
			final_prompt_to_global_ids[prompt] = [continuous_id]
	
	logger.info("Final normalized prompt to global IDs mapping:")
	for prompt, global_ids in final_prompt_to_global_ids.items():
		logger.info(f"  '{prompt}': {global_ids}")
	
	return final_prompt_to_global_ids, all_outputs_per_frame


def process_single_video(
	predictor, 
	video_path: str, 
	prompts: List[str], 
	num_sample_frames: int,
	output_dir: str,
	cfg,
	logger,
	video_events_file: str = None
) -> bool:
	"""Process a single video with given prompts, optionally with shot-level processing."""
	try:
		logger.info(f"Processing video: {video_path}")
		
		# Check if video exists
		if not os.path.exists(video_path):
			logger.error(f"Video file not found: {video_path}")
			return False
		
		# Load video frames for visualization
		logger.info("Loading video frames for visualization...")
		video_frames_for_vis = load_video_frames_for_vis(video_path)
		
		if len(video_frames_for_vis) == 0:
			logger.error("No frames found in video")
			return False
		
		total_frames = len(video_frames_for_vis)
		
		# Initialize shot processing variables
		shot_segments = []
		temp_dir = None
		shot_processing_enabled = cfg.get('shot_processing', {}).get('enabled', False)
		
		if shot_processing_enabled:
			# Get video events file path if not provided
			if not video_events_file:
				video_events_file = get_video_events_file_path(
					video_path, 
					cfg.shot_processing.get('video_events_base_dir')
				)
			
			# Check if video events file exists
			if not os.path.exists(video_events_file):
				logger.warning(f"Video events file not found: {video_events_file}")
				if cfg.shot_processing.get('fallback_to_full_video', True):
					logger.info("Falling back to full video processing")
					shot_processing_enabled = False
				else:
					logger.error("Shot processing enabled but no video events file found")
					return False
			else:
				# Read shot boundaries
				logger.info(f"Reading shot boundaries from: {video_events_file}")
				try:
					shot_boundaries = read_shot_boundaries(video_events_file)
					shot_segments = get_shot_segments(shot_boundaries, total_frames)
					logger.info(f"Found {len(shot_segments)} shots in video")
					
					# Setup temporary directory if needed
					if cfg.shot_processing.get('create_temp_videos', False):
						temp_dir = cfg.shot_processing.get('temp_dir', '/tmp/sam3_shot_clips')
						os.makedirs(temp_dir, exist_ok=True)
						logger.info(f"Using temporary directory: {temp_dir}")
				except Exception as e:
					logger.error(f"Failed to read shot boundaries file: {e}")
					logger.info("Falling back to full video processing")
					shot_processing_enabled = False
		
		# If no shot processing, treat entire video as one segment
		if not shot_processing_enabled:
			shot_segments = [(0, total_frames - 1)]
		
		# Process each shot (or single segment for non-shot processing)
		all_outputs_per_frame = {}
		global_obj_id_counter = 0
		
		# Initialize prompt to global IDs mapping
		prompt_to_global_ids = {}
		
		# Load Molmo points for refinement if enabled
		molmo_points_data = {}
		if cfg.get('points_refinement', {}).get('enabled', False):
			logger.info("Points refinement enabled, loading Molmo predictions...")
			try:
				# Get video ID from path (extract filename without extension)
				video_id = os.path.splitext(os.path.basename(video_path))[0]
				
				# Get video properties for coordinate normalization
				video_width, video_height, original_fps = get_video_properties(video_path)
				logger.info(f"Video properties: {video_width}x{video_height} at {original_fps} fps")
				
				molmo_points_data = load_molmo_points_for_video(
					molmo_predictions_path=cfg.points_refinement.molmo_predictions_json,
					video_id=video_id,
					prompts=prompts,
					video_width=video_width,
					video_height=video_height,
					cfg=cfg,
					logger=logger
				)
				
				if molmo_points_data:
					logger.info(f"Loaded Molmo points for {len(molmo_points_data)} prompts")
				else:
					logger.info("No Molmo points found for this video")
					
			except Exception as e:
				logger.error(f"Failed to load Molmo points: {str(e)}")
				molmo_points_data = {}
		
		try:
			for shot_idx, (start_frame, end_frame) in enumerate(shot_segments):
				if shot_processing_enabled:
					logger.info(f"Processing shot {shot_idx + 1}/{len(shot_segments)}: frames {start_frame}-{end_frame}")
				else:
					logger.info("Processing entire video")
				
				# Create temporary video for this shot if enabled
				temp_video_path = None
				if temp_dir and shot_processing_enabled:
					try:
						temp_video_path = create_temp_shot_video(
							video_path, start_frame, end_frame, temp_dir, shot_idx
						)
						logger.info(f"Created temporary video: {temp_video_path}")
					except Exception as e:
						return Exception(f"Failed to create temporary shot video: {e}")
				# Use temporary video if provided, otherwise use original
				working_video_path = temp_video_path if temp_video_path else video_path
				
				# Start inference session
				if shot_processing_enabled:
					logger.info(f"Starting inference session for shot {shot_idx}...")
				else:
					logger.info("Starting inference session...")
				
				response = predictor.handle_request(
					request=dict(type="start_session", resource_path=working_video_path)
				)
				session_id = response["session_id"]
				logger.info(f"Started session {session_id}")
				
				# Reset session
				_ = predictor.handle_request(
					request=dict(type="reset_session", session_id=session_id)
				)
				
				# Calculate frame indices for prompting
				if shot_processing_enabled:
					shot_length = end_frame - start_frame + 1
					if shot_length < num_sample_frames:
						# Use all frames in the shot
						if temp_video_path:
							sample_frame_indices = list(range(shot_length))
						else:
							sample_frame_indices = list(range(start_frame, end_frame + 1))
					else:
						# Sample frames within the shot
						if temp_video_path:
							sample_frame_indices = sorted(random.sample(range(shot_length), num_sample_frames))
						else:
							relative_indices = sorted(random.sample(range(shot_length), num_sample_frames))
							sample_frame_indices = [start_frame + idx for idx in relative_indices]
				else:
					# Original logic for full video
					n_frames = len(video_frames_for_vis)
					if n_frames < num_sample_frames:
						sample_frame_indices = list(range(n_frames))
					else:
							if num_sample_frames == 1:
								sample_frame_indices = [n_frames // 2]  # Middle frame
							else:
								sample_frame_indices = np.linspace(0, n_frames - 1, num_sample_frames, dtype=int).tolist()
				logger.info(f"Adding prompts to frames: {sample_frame_indices}")
				logger.info(f"Prompts: {prompts}")
				
				# Process each prompt for this shot/video
				for prompt in prompts:
					if shot_processing_enabled:
						logger.info(f"Shot {shot_idx} - Processing prompt '{prompt}' across sampled frames")
					else:
						logger.info(f"Processing prompt '{prompt}' across all sampled frames")
					
					# Reset session for each prompt
					_ = predictor.handle_request(
						request=dict(type="reset_session", session_id=session_id)
					)
					
					# Add current prompt to all sampled frames
					for frame_idx in sample_frame_indices:
						# For temp video, frame indices are relative to shot
						frame_idx = frame_idx
						logger.info(f"Adding prompt '{prompt}' to frame {frame_idx})")
						resp = predictor.handle_request(
							request=dict(
								type="add_prompt",
								session_id=session_id,
								frame_index=frame_idx,
								text=prompt,
							)
						)
						out = resp["outputs"]
						
						# Save individual frame visualization if enabled
						if cfg.output.save_frame_vis:
							safe_prompt = prompt.replace(" ", "_").replace("/", "_")
							if shot_processing_enabled:
								vis_path = os.path.join(
									output_dir,
									f"shot_{shot_idx:03d}_frame_{frame_idx:05d}_prompt_{safe_prompt}.{cfg.visualization.format}"
								)
								title = f"SAM 3 Shot {shot_idx}: prompt='{prompt}' (frame {frame_idx})"
							else:
								vis_path = os.path.join(
									output_dir, 
									f"frame_{frame_idx:05d}_prompt_{safe_prompt}.{cfg.visualization.format}"
								)
								title = f"SAM 3: prompt='{prompt}' (frame {frame_idx})"
							
							plt.close("all")
							
							# Use absolute frame index for visualization
							abs_frame_idx = frame_idx
							if shot_processing_enabled and temp_video_path:
								abs_frame_idx = start_frame + frame_idx
							
							if abs_frame_idx < len(video_frames_for_vis):
								visualize_frame_with_masks(
									frame_idx=abs_frame_idx,
									video_frames=video_frames_for_vis,
									masks_data={abs_frame_idx: out},
									title=title,
									figsize=cfg.visualization.figsize,
									save_path=vis_path
								)
								logger.info(f"Saved frame visualization: {vis_path}")
					
					# First propagate through the shot/video with text prompts only
					if shot_processing_enabled:
						logger.info(f"Shot {shot_idx} - Propagating masks for prompt '{prompt}' through the shot...")
					else:
						logger.info(f"Propagating masks for prompt '{prompt}' through the video...")
					
					prompt_outputs_per_frame = propagate_in_video(predictor, session_id)
					
					# Apply points refinement if enabled and points are available for this prompt
					if (cfg.get('points_refinement', {}).get('enabled', False) and 
						prompt in molmo_points_data and 
						len(molmo_points_data[prompt]['points']) > 0):
						
						logger.info(f"Applying points refinement for prompt '{prompt}' after initial propagation")
						points_data = molmo_points_data[prompt]
						points = points_data['points']
						labels = points_data['labels']
						detected_frames = points_data['frames']
						
						# Get object IDs from the initial propagation
						available_obj_ids = set()
						for frame_idx, frame_outputs in prompt_outputs_per_frame.items():
							if'out_obj_ids' in frame_outputs:
									available_obj_ids.update(frame_outputs['out_obj_ids'].tolist())
						
						if available_obj_ids:
							# Use the first available object ID for refinement
							refine_obj_id = sorted(available_obj_ids)[0]
							logger.info(f"Using object ID {refine_obj_id} for points refinement")
							
							# Determine which frames to refine
							if cfg.points_refinement.get('refine_on_detected_frames_only', True):
								# Only refine on frames where Molmo detected points
								refine_frames = detected_frames
							else:
								# Refine on all sampled frames
								refine_frames = sample_frame_indices
							
							# Apply points refinement on selected frames
							for refine_frame_idx in refine_frames:
								# Convert to relative frame index if using temp video
								frame_to_refine = refine_frame_idx
								if shot_processing_enabled and temp_video_path:
									# Check if this frame is within the current shot
									if start_frame <= refine_frame_idx <= end_frame:
										frame_to_refine = refine_frame_idx - start_frame
									else:
										continue  # Skip frames outside current shot
								elif shot_processing_enabled:
									# For shot processing without temp video, check frame bounds
									if not (start_frame <= refine_frame_idx <= end_frame):
										continue
								
								# Find corresponding points for this frame
								frame_points = []
								frame_labels = []
								
								if refine_frame_idx in detected_frames:
									# Use specific points for this frame
									point_idx = detected_frames.index(refine_frame_idx)
									if point_idx < len(points):
										frame_points = points[point_idx:point_idx+1]  # Single point
										frame_labels = labels[point_idx:point_idx+1]
								else:
									# Use all available points for this prompt
									frame_points = points
									frame_labels = labels
								
								if len(frame_points) > 0:
									logger.info(f"Refining frame {refine_frame_idx} with {len(frame_points)} points for obj_id {refine_obj_id}")
									
									# Apply point refinement with the object ID
									refine_resp = predictor.handle_request(
										request=dict(
											type="add_prompt",
											session_id=session_id,
											frame_index=frame_to_refine,
											points=frame_points,
											point_labels=frame_labels,
											obj_id=refine_obj_id,
										)
									)
									logger.info(f"Successfully refined frame {refine_frame_idx}")
										
							
							# Propagate again after points refinement
							logger.info(f"Re-propagating after points refinement for prompt '{prompt}'")
							prompt_outputs_per_frame = propagate_in_video(predictor, session_id)
							
						else:
							logger.warning(f"No object IDs found for prompt '{prompt}', skipping points refinement")
					
					# Adjust frame indices to be absolute if using temp video
					if shot_processing_enabled and temp_video_path:
						adjusted_outputs = {}
						for rel_frame_idx, frame_outputs in prompt_outputs_per_frame.items():
							abs_frame_idx = start_frame + rel_frame_idx
							adjusted_outputs[abs_frame_idx] = frame_outputs
						prompt_outputs_per_frame = adjusted_outputs
					
					# Get the maximum object ID from this prompt's results
					prompt_max_obj_id = -1
					prompt_global_ids = []  # Track global IDs for this prompt
					for frame_outputs in prompt_outputs_per_frame.values():
						if isinstance(frame_outputs, dict) and 'out_obj_ids' in frame_outputs:
							if frame_outputs['out_obj_ids'].size > 0:
								if cfg.text_prompt_setting == "one_object_per_prompt":
									prompt_max_obj_id = max(prompt_max_obj_id, frame_outputs['out_obj_ids'][0])
								else:
									prompt_max_obj_id = max(prompt_max_obj_id, max(frame_outputs['out_obj_ids']))
					
					# Initialize prompt in mapping if not exists
					if prompt not in prompt_to_global_ids:
						prompt_to_global_ids[prompt] = []
					
					# Merge outputs with global object ID management
					for frame_idx, frame_outputs in prompt_outputs_per_frame.items():
						if frame_idx not in all_outputs_per_frame:
							all_outputs_per_frame[frame_idx] = {}
						
						if isinstance(frame_outputs, dict):
							for key, value in frame_outputs.items():
								if isinstance(value, dict) and value is not None:
									if key not in all_outputs_per_frame[frame_idx]:
										all_outputs_per_frame[frame_idx][key] = {}
									for stat_key, stat_value in value.items():
										stat_value = [stat_value]
										if stat_value and cfg.text_prompt_setting == "one_object_per_prompt" and stat_key=="num_obj_tracked":
											stat_value = [1]
										if stat_key not in all_outputs_per_frame[frame_idx][key]:
											all_outputs_per_frame[frame_idx][key][stat_key] = list(stat_value)
										else:
											all_outputs_per_frame[frame_idx][key][stat_key].extend(list(stat_value))
								
								if isinstance(value, list) and value is not None:
									if key == "out_probs":
										if cfg.text_prompt_setting == "one_object_per_prompt":
											value = list(value[0])
										if key not in all_outputs_per_frame[frame_idx]:
											all_outputs_per_frame[frame_idx][key] = list(value)
										else:
											all_outputs_per_frame[frame_idx][key].extend(list(value))
								
								if isinstance(value, np.ndarray) and value.size > 0 and key != "out_obj_ids":
									if cfg.text_prompt_setting == "one_object_per_prompt":
										new_values = [value[0]]
									else:
										new_values = [v for v in value]
									if key not in all_outputs_per_frame[frame_idx]:
										all_outputs_per_frame[frame_idx][key] = new_values
									else:
										all_outputs_per_frame[frame_idx][key].extend(new_values)
								
								elif key == "out_obj_ids" and value.size > 0:
									if cfg.text_prompt_setting == "one_object_per_prompt":
										new_obj_ids = [value[0] + global_obj_id_counter]
									else:
										new_obj_ids = [obj_id + global_obj_id_counter for obj_id in value]
									
									# Track global IDs for this prompt
									prompt_global_ids.extend(new_obj_ids)
									
									if key not in all_outputs_per_frame[frame_idx]:
										all_outputs_per_frame[frame_idx][key] = new_obj_ids
									else:
										all_outputs_per_frame[frame_idx][key].extend(new_obj_ids)
					
					# Update prompt to global IDs mapping (ensure unique IDs)
					unique_prompt_global_ids = list(set(int(gid) for gid in prompt_global_ids)) 
					prompt_to_global_ids[prompt].extend(unique_prompt_global_ids)
					prompt_to_global_ids[prompt] = list(set(prompt_to_global_ids[prompt]))  # Remove duplicates
					
					# Update global object ID counter for next prompt
					if prompt_max_obj_id >= 0:
						global_obj_id_counter += prompt_max_obj_id + 1
						if shot_processing_enabled:
							logger.info(f"Shot {shot_idx} - Updated global object ID counter to {global_obj_id_counter}")
							logger.info(f"Shot {shot_idx} - Prompt '{prompt}' mapped to global IDs: {unique_prompt_global_ids}")
						else:
							logger.info(f"Updated global object ID counter to {global_obj_id_counter}")
							logger.info(f"Prompt '{prompt}' mapped to global IDs: {unique_prompt_global_ids}")
				
				# Close session for this shot
				_ = predictor.handle_request(
					request=dict(type="close_session", session_id=session_id)
				)
				
				if shot_processing_enabled:
					logger.info(f"Successfully processed shot {shot_idx}")
		
		finally:
			# Cleanup temporary files
			if temp_dir and cfg.shot_processing.get('cleanup_temp_files', True):
				cleanup_temp_directory(temp_dir, logger)
		
		# Create outputs structure (same format as original)
		outputs_per_frame = {'preds': all_outputs_per_frame, 'text_prompts': prompts, 'prompt_to_global_ids': prompt_to_global_ids}
		
		# Log final prompt to global IDs mapping
		logger.info("Final prompt to global IDs mapping:")
		for prompt, global_ids in prompt_to_global_ids.items():
			logger.info(f"  '{prompt}': {global_ids}")
		
		# Normalize object IDs using dedicated function
		prompt_to_global_ids, all_outputs_per_frame = normalize_object_ids(all_outputs_per_frame, prompt_to_global_ids, logger)
		
		# Save propagated visualizations if enabled
		if cfg.output.save_propagated_vis:
			n_frames_out = len(outputs_per_frame['preds'])
			if n_frames_out > 0:
				if cfg.output.sample_input_prompt_frames_for_vis:
					if shot_processing_enabled:
						# Sample from shot boundaries and middle frames
						sample_idxs = []
						for start_frame, end_frame in shot_segments[:5]:  # Limit to first 5 shots
							sample_idxs.extend([start_frame, (start_frame + end_frame) // 2, end_frame])
						sample_idxs = list(set(sample_idxs))  # Remove duplicates
					else:
						# Use original sampling logic
						sample_idxs = random.sample(list(all_outputs_per_frame.keys()), min(num_sample_frames, n_frames_out))
				else:
					sample_idxs = [0, total_frames // 2, max(0, total_frames - 1)]
				
				plt.close("all")
				for idx in sample_idxs:
					if idx < len(video_frames_for_vis) and idx in all_outputs_per_frame:
						if shot_processing_enabled:
							out_path = os.path.join(
								output_dir,
								f"frame_{idx:05d}_shot_propagated.{cfg.visualization.format}"
							)
							title = f"SAM 3 shot-based propagated outputs (frame {idx})"
						else:
							out_path = os.path.join(
								output_dir, 
								f"frame_{idx:05d}_propagated.{cfg.visualization.format}"
							)
							title = f"SAM 3 propagated outputs (frame {idx})"
						
						visualize_frame_with_masks(
							frame_idx=idx,
							video_frames=video_frames_for_vis,
							masks_data=all_outputs_per_frame,
							title=title,
							figsize=cfg.visualization.figsize,
							save_path=out_path
						)
						logger.info(f"Saved propagated visualization: {out_path}")
		
		# Save raw outputs if enabled (same format as original)
		if cfg.output.save_raw_outputs:
			raw_save = {}
			
			# Store video metadata
			raw_save["video_metadata"] = {
				"video_path": video_path,
				"video_id": getattr(cfg, 'current_video_id', None),  # Will be set externally
				"video_index": getattr(cfg, 'current_video_index', None),  # Will be set externally
				"total_frames": total_frames,
				"processed_frames": list(all_outputs_per_frame.keys())
			}
			
			# Store frame-wise predictions with bounding boxes
			for frame_idx, out in outputs_per_frame['preds'].items():
				if isinstance(out, dict):
					# Extract masks, object IDs, and bounding boxes
					masks_list = []
					obj_ids_list = []
					bboxes_list = []
					
					if 'out_binary_masks' in out and out['out_binary_masks'] is not None:
						masks_list = out['out_binary_masks']
					if 'out_obj_ids' in out and out['out_obj_ids'] is not None:
						obj_ids_list = out['out_obj_ids']
					
					# Extract or compute bounding boxes from masks
					if masks_list:
						for mask in masks_list:
							if mask is not None and mask.sum() > 0:
								# Compute bounding box from mask
								coords = np.where(mask)
								if len(coords[0]) > 0:
									y_min, y_max = coords[0].min(), coords[0].max()
									x_min, x_max = coords[1].min(), coords[1].max()
									# Convert to [x, y, width, height] format
									bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
									bboxes_list.append(bbox)
								else:
									bboxes_list.append([0.0, 0.0, 0.0, 0.0])
							else:
								bboxes_list.append([0.0, 0.0, 0.0, 0.0])
					
					raw_save[f"frame_{frame_idx}"] = {
						"masks": masks_list if masks_list else None,
						"obj_ids": obj_ids_list if obj_ids_list else None,
						"bboxes": bboxes_list if bboxes_list else None
					}
				else:
					raw_save[f"frame_{frame_idx}"] = {
						"masks": None,
						"obj_ids": None,
						"bboxes": None
					}
			
			# Include prompt to global IDs mapping in raw outputs
			raw_save["prompt_to_global_ids"] = prompt_to_global_ids
			raw_save["text_prompts"] = prompts
			
			# Save as pickle since masks have different shapes and can't be stored as homogeneous arrays in NPZ
			pkl_path = os.path.join(output_dir, "sam3_raw_outputs.pkl")
			with open(pkl_path, "wb") as f:
				pickle.dump(raw_save, f)
			logger.info(f"Saved raw outputs to {pkl_path}")
			
			mapping_path = os.path.join(output_dir, "prompt_to_global_ids.json")
			with open(mapping_path, "w") as f:
				json.dump({
					"prompt_to_global_ids": prompt_to_global_ids,
					"text_prompts": list(prompts),
					"video_metadata": raw_save["video_metadata"]
				}, f, indent=2)
			logger.info(f"Saved prompt to global IDs mapping to {mapping_path}")
			
			# Generate per-frame annotation files
			generate_per_frame_annotations(outputs_per_frame['preds'], prompt_to_global_ids, output_dir, logger)
		
		# Create track visualization video if enabled
		if cfg.get('video_tracks', {}).get('enabled', False) and cfg.get('output', {}).get('save_track_video', False):
			logger.info("Creating track visualization video...")
			try:
				video_file_path = create_track_visualization_video(
					video_frames_for_vis=video_frames_for_vis,
					outputs_per_frame=outputs_per_frame['preds'],
					output_dir=output_dir,
					cfg=cfg,
					logger=logger,
					prompt_to_global_ids=prompt_to_global_ids,
					video_path=video_path
				)
				logger.info(f"Saved track visualization video: {video_file_path}")
			except Exception as e:
				logger.error(f"Failed to create track visualization video: {str(e)}")
				if logger.level <= logging.DEBUG:
					logger.error(traceback.format_exc())
		
		logger.info(f"Successfully processed video: {video_path}")
		return True
		
	except Exception as e:
		logger.error(f"Error processing video {video_path}: {str(e)}")
		if logger.level <= logging.DEBUG:
			logger.error(traceback.format_exc())
		return False
