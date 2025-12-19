
import sys
from omegaconf import OmegaConf
import os
import logging
import numpy as np
import json
import torch
from typing import List, Dict, Tuple, Optional
import glob
import cv2
import subprocess
import shutil
from pathlib import Path



def load_config(config_path="config.yaml"):
	"""Load configuration from a YAML file and apply CLI overrides.

	CLI overrides should be provided as key=value pairs after the script name,
	e.g. python script.py device=cuda save_memory=False
	"""
	if not os.path.exists(config_path):
		raise FileNotFoundError(f"Config file not found: {config_path}")
		
	cfg = OmegaConf.load(config_path)
	if len(sys.argv) > 1:
		# Filter out --config argument and its value
		filtered_args = []
		skip_next = False
		for i, arg in enumerate(sys.argv[1:]):
			if skip_next:
				skip_next = False
				continue
			if arg == '--config':
				skip_next = True
				continue
			filtered_args.append(arg)
		
		if filtered_args:
			cli_cfg = OmegaConf.from_cli(filtered_args)
			cfg = OmegaConf.merge(cfg, cli_cfg)

	return cfg

def read_shot_boundaries(shot_file):
    """
    Reads shot boundary frame indices from a .vdbens file.

    Args:
        shot_file (str): Path to the .vdbens file.

    Returns:
        list: List of frame indices marking the start of each shot.
    """
    with open(shot_file, "r") as f:
        lines = f.readlines()

    # Skip the first line and parse the remaining lines
    shot_boundaries = []
    for line in lines[1:]:  # Skip the first line
        parts = line.split()
        if parts:  # Ensure the line is not empty
            shot_boundaries.append(int(parts[0]))  # First integer is the shot number

    # If 0 not present in the list, append it to indicate the start of the first shot at the start of the list
    if len(shot_boundaries) == 0 or 0 not in shot_boundaries:
        shot_boundaries.insert(0, 0)

    return shot_boundaries

def get_video_events_file_path(video_path: str, video_events_base_dir: Optional[str] = None) -> str:
    """
    Get the corresponding .videvents file path for a video file.
    
    Args:
        video_path (str): Path to the video file
        video_events_base_dir (Optional[str]): Base directory for .videvents files.
                                             If None, looks in same directory as video.
    
    Returns:
        str: Path to the corresponding .videvents file
    """
    video_name = Path(video_path).stem  # Get filename without extension
    if video_events_base_dir:
        events_file = os.path.join(video_events_base_dir, video_name, video_name, f"{video_name}.videvents")
    return events_file


def get_shot_segments(shot_boundaries: List[int], total_frames: int) -> List[Tuple[int, int]]:
    """
    Convert shot boundaries to shot segments (start, end) pairs.
    
    Args:
        shot_boundaries (List[int]): List of shot starting frame indices
        total_frames (int): Total number of frames in the video
    
    Returns:
        List[Tuple[int, int]]: List of (start_frame, end_frame) tuples for each shot
    """
    if not shot_boundaries:
        return [(0, total_frames - 1)]
    
    segments = []
    for i in range(len(shot_boundaries)):
        start_frame = shot_boundaries[i]
        if i + 1 < len(shot_boundaries):
            end_frame = shot_boundaries[i + 1] - 1
        else:
            end_frame = total_frames - 1
        
        if start_frame <= end_frame:
            segments.append((start_frame, end_frame))
    
    return segments


def create_temp_shot_video(
    video_path: str,
    start_frame: int,
    end_frame: int,
    temp_dir: str,
    shot_idx: int,
) -> str:
    """
    Create a temporary video for a shot using exact frame indices.

    Frames are selected explicitly (no time-based seeking) and encoded
    using the same FPS as the source video.
    """

    os.makedirs(temp_dir, exist_ok=True)

    video_name = Path(video_path).stem
    temp_video_path = os.path.join(
        temp_dir, f"{video_name}_shot_{shot_idx:03d}.mp4"
    )

    # Read FPS from original video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Frame-accurate selection
    # select='between(n,start,end)' keeps only desired frames
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"select=between(n\\,{start_frame}\\,{end_frame})",
        "-vsync", "0",
        "-r", str(fps),
        "-c:v", "libx264",          # same *type* of codec
        "-pix_fmt", "yuv420p",
        temp_video_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return temp_video_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to create temporary shot video:\n{e.stderr.decode()}"
        )


def cleanup_temp_directory(temp_dir: str, logger=None) -> None:
    """
    Clean up temporary directory and all its contents.
    
    Args:
        temp_dir (str): Path to temporary directory to clean up
        logger: Optional logger for reporting cleanup status
    """
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            if logger:
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
            else:
                print(f"Warning: Failed to clean up temporary directory {temp_dir}: {e}")


def setup_logging(cfg):
	"""Setup logging based on configuration."""
	from sam3.logger import get_logger
	
	logging.basicConfig(
		level=getattr(logging, cfg.logging.level),
		format='%(asctime)s - %(levelname)s - %(message)s',
		handlers=[
			logging.StreamHandler(),
		]
	)
	logger = logging.getLogger(__name__)

	# # Use SAM3's built-in logger with the configured level
	# log_level = getattr(logging, cfg.logging.level)
	# logger = get_logger(__name__, level=log_level)
	
	if cfg.logging.save_to_file:
		os.makedirs(cfg.output.base_dir, exist_ok=True)
		log_file_path = os.path.join(cfg.output.base_dir, cfg.logging.log_file)
		file_handler = logging.FileHandler(log_file_path)
		file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
		logger.addHandler(file_handler)
	
	return logger

def reencode_to_h264(input_path: str, output_path: str, fps: int):
    import subprocess

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        output_path,
    ]

    subprocess.run(cmd, check=True)


def create_track_visualization_video(
	video_frames_for_vis: List,
	outputs_per_frame: Dict,
	output_dir: str,
	cfg,
	logger,
	prompt_to_global_ids: Dict = None,
	video_path: str = None
) -> str:
	"""Create a video showing object tracks overlaid on the original frames."""
	import cv2

	# Create reverse mapping from global ID to prompt
	global_id_to_prompt = {}
	if prompt_to_global_ids:
		for prompt, global_ids in prompt_to_global_ids.items():
			for global_id in global_ids:
				global_id_to_prompt[global_id] = prompt

	# Get video track configuration
	video_cfg = cfg.get('video_tracks', {})
	output_format = video_cfg.get('output_format', 'mp4')
	
	# Get FPS - use original video FPS if config fps is null, otherwise use config value
	config_fps = video_cfg.get('fps', None)
	if config_fps is None and video_path is not None:
		try:
			fps = get_video_fps(video_path)
			logger.info(f"Using original video FPS: {fps}")
		except Exception as e:
			logger.warning(f"Could not detect video FPS: {e}, using default 24 fps")
			fps = 24.0
	elif config_fps is not None:
		fps = config_fps
		logger.info(f"Using config FPS: {fps}")
	else:
		fps = 24.0  # Default fallback
		logger.info(f"Using default FPS: {fps}")
	
	# Visualization options
	show_masks = video_cfg.get('show_masks', True)
	show_object_ids = video_cfg.get('show_object_ids', True)
	show_bboxes = video_cfg.get('show_bboxes', True)
	mask_alpha = video_cfg.get('mask_alpha', 0.6)
	bbox_thickness = video_cfg.get('bbox_thickness', 2)
	text_size = video_cfg.get('text_size', 0.5)
	
	# Prepare output video path
	video_output_path = os.path.join(output_dir, f"track_visualization.{output_format}")
	
	# Get video dimensions from first frame
	first_frame = video_frames_for_vis[0] if isinstance(video_frames_for_vis[0], np.ndarray) else cv2.imread(video_frames_for_vis[0])
	if first_frame is None:
		raise ValueError("Could not load first frame")
	
	height, width = first_frame.shape[:2]
	
	# Set up video writer
	fourcc = cv2.VideoWriter_fourcc(*'mp4v') if output_format == 'mp4' else cv2.VideoWriter_fourcc(*'XVID')
	video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
	
	if not video_writer.isOpened():
		raise RuntimeError(f"Could not open video writer for {video_output_path}")
	
	logger.info(f"Video writer initialized: {width}x{height} at {fps} fps")
	
	# Generate colors for different object IDs
	colors = {
		0: (255, 0, 0),    # Red
		1: (0, 255, 0),    # Green
		2: (0, 0, 255),    # Blue
		3: (255, 255, 0),  # Cyan
		4: (255, 0, 255),  # Magenta
		5: (0, 255, 255),  # Yellow
		6: (128, 0, 128),  # Purple
		7: (255, 165, 0),  # Orange
		8: (0, 128, 0),    # Dark Green
		9: (128, 128, 0),  # Olive
	}
	
	logger.info(f"Creating track visualization video with {len(video_frames_for_vis)} frames")
	
	frames_written = 0
	# set all keys and values for each frame_idx in outputs_per_frame to corresponding 
	key_to_type_dict = {
		'frame_stats' : {}, # dict
		'out_probs': [], # list
		'out_obj_ids': [], # list
		'out_binary_masks': np.array([]), # np.ndarray
		'out_boxes_xywh': np.array([]), # np.ndarray
	}
	for frame_idx in range(len(video_frames_for_vis)):
		if frame_idx not in outputs_per_frame:
			outputs_per_frame[frame_idx] = {key: key_to_type_dict[key] for key in key_to_type_dict}
		else:
			for key in key_to_type_dict:
				if key not in outputs_per_frame[frame_idx]:
					outputs_per_frame[frame_idx][key] = key_to_type_dict[key]
	# Process each frame
	for frame_idx in range(len(video_frames_for_vis)):
		# Load frame
		if isinstance(video_frames_for_vis[frame_idx], np.ndarray):
			frame = video_frames_for_vis[frame_idx].copy()
			# Convert RGB to BGR for OpenCV
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		else:
			frame = cv2.imread(video_frames_for_vis[frame_idx])
		
		if frame is None:
			continue
		
		# Get outputs for this frame if available
		if frame_idx in outputs_per_frame:
			
			masks = list(outputs_per_frame[frame_idx]['out_binary_masks'])
			obj_ids = list(outputs_per_frame[frame_idx]['out_obj_ids'])
			
			# Process each detected object
			for i, (mask, obj_id) in enumerate(zip(masks, obj_ids)):
				if mask is None:
					continue
					
				# Convert mask to numpy array if needed
				if hasattr(mask, 'cpu'):
					mask = mask.cpu().numpy()
				mask = mask.astype(np.uint8)
				
				# Ensure mask is 2D
				if mask.ndim == 1:
					# Skip 1D masks as we can't determine proper 2D shape
					continue
				elif mask.ndim > 2:
					# If mask has extra dimensions, take the first 2D slice
					mask = mask.squeeze()  # Remove singleton dimensions
					if mask.ndim != 2:
						continue  # Skip if still not 2D after squeeze
				
				# Get color for this object (handle both string and int obj_ids)
				obj_id_int = int(obj_id) if isinstance(obj_id, str) else obj_id
				color = colors.get(obj_id_int % len(colors), (255, 255, 255))
				
				# Show mask overlay
				if show_masks and mask.shape[:2] == frame.shape[:2]:
					colored_mask = np.zeros_like(frame)
					colored_mask[mask > 0] = color
					frame = cv2.addWeighted(frame, 1 - mask_alpha, colored_mask, mask_alpha, 0)
				
				# Calculate object center and bounding box
				if np.any(mask > 0):
					coords = np.where(mask > 0)
					if len(coords) == 2 and len(coords[0]) > 0 and len(coords[1]) > 0:
						y_coords, x_coords = coords
						
						# Show bounding box
						if show_bboxes:
							min_x, max_x = np.min(x_coords), np.max(x_coords)
							min_y, max_y = np.min(y_coords), np.max(y_coords)
							cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, bbox_thickness)
						# Show object ID and prompt (top-right, outside the box)
						if show_object_ids:
							# Get prompt for this object ID
							prompt_text = global_id_to_prompt.get(obj_id, "")
							if prompt_text:
								label = f"ID:{obj_id} | {prompt_text[:20]}"  # Limit prompt text to 20 chars
							else:
								label = f"ID:{obj_id}"
							
							(text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_size, 2)
							cv2.rectangle(frame, (max_x + 5, min_y),
											(max_x + 5 + text_width, min_y + text_height + 5), color, -1)
							cv2.putText(frame, label, (max_x + 5, min_y + text_height),
										cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), 2)

		# Write frame to video (always write the frame, regardless of trails)
		video_writer.write(frame)
		frames_written += 1
	
	# Release video writer
	video_writer.release()
	
	# To see the file in vscode itself.
	h264_path = video_output_path.replace(".mp4", "_h264.mp4")
	reencode_to_h264(video_output_path, h264_path, fps)
	shutil.move(h264_path, video_output_path)
	logger.info(f"Video creation completed: {frames_written} frames written to {video_output_path}")
	
	return video_output_path



def visualize_frame_with_masks(frame_idx: int, video_frames: List, masks_data: Dict, title: str = "", figsize: Tuple[int, int] = (12, 8), save_path: str = None) -> None:
	"""Visualize a frame with overlay masks.
	
	Args:
		frame_idx: Index of the frame to visualize
		video_frames: List of video frames (RGB arrays or file paths)
		masks_data: Dictionary containing masks and obj_ids for the frame
		title: Title for the plot
		figsize: Figure size tuple
		save_path: Optional path to save the figure
	"""
	import matplotlib.pyplot as plt
	import matplotlib.colors as mcolors
	
	# Load the frame
	if isinstance(video_frames[frame_idx], np.ndarray):
		frame = video_frames[frame_idx]
	else:
		frame = cv2.imread(video_frames[frame_idx])
		if frame is not None:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	
	if frame is None:
		return
	
	fig, ax = plt.subplots(figsize=figsize)
	ax.imshow(frame)
	
	# Get masks for this frame
	if frame_idx in masks_data and 'out_binary_masks' in masks_data[frame_idx]:
		masks = list(masks_data[frame_idx]['out_binary_masks'])
		obj_ids = list(masks_data[frame_idx].get('obj_ids', range(len(masks))))
		
		# Define colors for different objects
		colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple']
		
		for i, (mask, obj_id) in enumerate(zip(masks, obj_ids)):
			if mask is not None and np.any(mask):
				# Convert mask to proper format if needed
				if hasattr(mask, 'cpu'):
					mask = mask.cpu().numpy()
				mask = mask.astype(bool)
				
				# Ensure mask is 2D
				if mask.ndim == 1:
					# Skip 1D masks as we can't determine proper 2D shape
					continue
				elif mask.ndim > 2:
					# If mask has extra dimensions, take the first 2D slice
					mask = mask.squeeze()  # Remove singleton dimensions
					if mask.ndim != 2:
						continue  # Skip if still not 2D after squeeze
				
				# Ensure mask matches frame dimensions
				if mask.shape[:2] != frame.shape[:2]:
					continue
				
				# Create colored mask overlay
				color = colors[i % len(colors)]
				colored_mask = np.zeros((*mask.shape, 4))
				color_rgb = mcolors.to_rgba(color, alpha=0.6)
				colored_mask[mask] = color_rgb
				
				ax.imshow(colored_mask)
				
				# Add object ID label at mask center
				coords = np.where(mask)
				if len(coords) == 2 and len(coords[0]) > 0 and len(coords[1]) > 0:
					y_coords, x_coords = coords
					center_x = np.mean(x_coords)
					center_y = np.mean(y_coords)
					ax.text(center_x, center_y, f'ID:{obj_id}', 
						color='white', fontweight='bold', fontsize=12,
						ha='center', va='center',
						bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
	
	ax.set_title(title if title else f"Frame {frame_idx}")
	ax.axis('off')
	
	if save_path:
		plt.savefig(save_path, dpi=150, bbox_inches='tight')
	else:
		plt.show()



def load_video_frames_for_vis(video_path: str) -> List:
	"""Return a list of RGB frames (ndarray) for visualization.

	Accepts either a folder of JPEGs (named like <frame_index>.jpg) or an MP4 file.
	"""
	if isinstance(video_path, str) and video_path.endswith(".mp4"):
		cap = cv2.VideoCapture(video_path)
		frames = []
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		cap.release()
		return frames
	else:
		imgs = glob.glob(os.path.join(video_path, "*.jpg"))
		try:
			imgs.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
		except ValueError:
			imgs.sort()
		return imgs


def load_molmo_points_for_video(molmo_predictions_path: str, video_id: str, prompts: List[str], 
								video_width: int, video_height: int, cfg, logger) -> Dict:
	"""
	Load Molmo point predictions for a specific video and prompts.
	
	Args:
		molmo_predictions_path: Path to the Molmo predictions JSON file
		video_id: ID of the video to load points for
		prompts: List of prompts to match against
		video_width: Width of the video (for normalization)
		video_height: Height of the video (for normalization)
		cfg: Configuration object with points_refinement settings
		logger: Logger instance
	
	Returns:
		Dictionary mapping prompts to their points data:
		{
			prompt: {
				'points': torch.Tensor,     # Shape: [n_points, 2] in relative coords
				'labels': torch.Tensor,     # Shape: [n_points] with values 1 (positive)
				'frames': List[int]         # Frame indices where points were detected
			}
		}
	"""
	try:
		# Load Molmo predictions
		with open(molmo_predictions_path, 'r') as f:
			molmo_data = json.load(f)
		
		# Check if video exists in predictions
		if video_id not in molmo_data:
			logger.warning(f"Video {video_id} not found in Molmo predictions")
			return {}
		
		video_data = molmo_data[video_id].get('molmo', {})
		if not video_data:
			logger.warning(f"No molmo data found for video {video_id}")
			return {}
		
		points_data = {}
		
		for prompt in prompts:
			# Find matching prompt in Molmo data (exact match or substring match)
			matched_prompt = None
			for molmo_prompt in video_data.keys():
				if prompt.lower() == molmo_prompt.lower() or prompt.lower() in molmo_prompt.lower():
					matched_prompt = molmo_prompt
					break
			
			if matched_prompt is None:
				logger.info(f"No Molmo points found for prompt '{prompt}' in video {video_id}")
				continue
			
			prompt_data = video_data[matched_prompt]
			points_list = prompt_data.get('Points', [])
			frames_list = prompt_data.get('Frames', [])
			
			if not points_list or not frames_list:
				logger.info(f"Empty points or frames for prompt '{prompt}' in video {video_id}")
				continue
			
			# Limit number of points if specified
			max_points = cfg.points_refinement.get('max_points_per_prompt', None)
			if max_points and len(points_list) > max_points:
				logger.info(f"Limiting points for '{prompt}' from {len(points_list)} to {max_points}")
				points_list = points_list[:max_points]
				frames_list = frames_list[:max_points]
			
			# Convert to torch tensors
			points_array = np.array(points_list, dtype=np.float32)
			
			# Normalize coordinates to relative (0-1) if enabled
			if cfg.points_refinement.get('normalize_coordinates', True):
				points_array[:, 0] = points_array[:, 0] / video_width   # x coordinates
				points_array[:, 1] = points_array[:, 1] / video_height  # y coordinates
				
				# Clamp to [0, 1] range to handle any edge cases
				points_array = np.clip(points_array, 0.0, 1.0)
			
			points_tensor = torch.tensor(points_array, dtype=torch.float32)
			
			# Create labels (all positive points if use_positive_points_only is True)
			if cfg.points_refinement.get('use_positive_points_only', True):
				labels = torch.ones(len(points_list), dtype=torch.int32)
			else:
				# Could implement negative points logic here if needed
				labels = torch.ones(len(points_list), dtype=torch.int32)
			
			points_data[prompt] = {
				'points': points_tensor,
				'labels': labels,
				'frames': frames_list
			}
			
			logger.info(f"Loaded {len(points_list)} points for prompt '{prompt}' across frames {frames_list}")
		
		return points_data
	
	except Exception as e:
		logger.error(f"Failed to load Molmo points for video {video_id}: {str(e)}")
		return {}


def get_video_dimensions(video_path: str) -> Tuple[int, int]:
	"""
	Get video dimensions (width, height) from video file.
	
	Args:
		video_path: Path to the video file
		
	Returns:
		Tuple of (width, height)
	"""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise ValueError(f"Could not open video: {video_path}")
	
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	cap.release()
	
	return width, height


def get_video_fps(video_path: str) -> float:
	"""
	Get video frame rate (FPS) from video file.
	
	Args:
		video_path: Path to the video file
		
	Returns:
		Frame rate as float
	"""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise ValueError(f"Could not open video: {video_path}")
	
	fps = cap.get(cv2.CAP_PROP_FPS)
	cap.release()
	
	# Fallback to 30 FPS if unable to detect
	if fps <= 0:
		fps = 30.0
	
	return fps


def get_video_properties(video_path: str) -> Tuple[int, int, float]:
	"""
	Get video properties (width, height, fps) from video file efficiently.
	
	Args:
		video_path: Path to the video file
		
	Returns:
		Tuple of (width, height, fps)
	"""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise ValueError(f"Could not open video: {video_path}")
	
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS)
	cap.release()
	
	# Fallback to 30 FPS if unable to detect
	if fps <= 0:
		fps = 30.0
	
	return width, height, fps

