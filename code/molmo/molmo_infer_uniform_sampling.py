from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import os
import regex as re
import torch
import numpy as np
from tqdm import tqdm
from utils import (
    save_results,
    load_finished_videos,
    save_finished_videos,
    visualize_video_results,
    read_shot_boundaries,
    VideoLoader,
    save_frame_to_tmp,
    load_config,
    get_unique_args_for_grounding,
)

def load_model_and_processor(model_name, data_type="bfloat16", device_map='auto'):
    if data_type == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = 'auto'

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    return model, processor


def parse_coordinates_universal(generated_text, image_width, image_height):
    points = []
    for line in generated_text.splitlines():
        if "There are none." in line:
            points.append(None)
        else:
            single_match = re.search(r'<point x="([\d.]+)" y="([\d.]+)"', line)
            if single_match:
                x = float(single_match.group(1)) * image_width / 100
                y = float(single_match.group(2)) * image_height / 100
                points.append((x, y))
            else:
                multiple_match = re.search(r'<points x1="([\d.]+)" y1="([\d.]+)"', line)
                if multiple_match:
                    x1 = float(multiple_match.group(1)) * image_width / 100
                    y1 = float(multiple_match.group(2)) * image_height / 100
                    points.append((x1, y1))
                else:
                    points.append(None)
    return points


def point_with_molmo(model, processor, image_paths, text):
    images = [Image.open(image_path) for image_path in image_paths]
    width, height = images[0].size
    text = f"Point {text}"
    inputs = processor.process(images=images, text=text)
    inputs["images"] = inputs["images"].to(torch.bfloat16)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
        )
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return parse_coordinates_universal(generated_text, width, height)


def get_uniform_sample_frames(total_frames, num_sample_frames):
    """
    Generate uniformly sampled frame indices.
    
    Args:
        total_frames (int): Total number of frames in the video.
        num_sample_frames (int): Number of frames to sample uniformly.
    
    Returns:
        list: List of frame indices to sample.
    """
    if num_sample_frames >= total_frames:
        return list(range(total_frames))
    
    # Create uniform spacing
    step = total_frames / num_sample_frames
    frame_indices = [int(i * step) for i in range(num_sample_frames)]
    
    # Ensure we don't exceed the total frame count
    frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]
    
    return frame_indices


def get_total_video_frames(video_path):
    """
    Get the total number of frames in a video.
    
    Args:
        video_path (str): Path to the video file.
    
    Returns:
        int: Total number of frames in the video.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return total_frames


def process_videos_uniform_sampling_molmo(config, unique_args, model, processor):
    """
    Process videos using uniform sampling with Molmo model.
    
    Args:
        config (dict): Configuration settings.
        unique_args (dict): Unique arguments for grounding, keyed by video ID.
        model: Loaded Molmo model.
        processor: Loaded Molmo processor.
    """
    all_video_molmo_preds = {}

    base_mp4_folder = config.base_mp4_folder
    base_tmp_dir = config.base_tmp_dir
    num_sample_frames = config.num_sample_frames

    if config.follow_input_vid_list:
        # Read video IDs from the input file
        with open(config.input_vid_file, "r") as f:
            videos_to_process = sorted([line.strip() for line in f.readlines()])
    else:
        # Use the current approach to list videos in base dirs
        if config.save_interval > 0:
            finished_videos = load_finished_videos(config.finished_file)
            videos = sorted(list(unique_args.keys()))[config.video_range[0]:config.video_range[1]]
            videos_to_process = [v for v in videos if v not in finished_videos]
        else:
            videos_to_process = sorted(list(unique_args.keys()))[config.video_range[0]:config.video_range[1]]
    
    print(f"Total videos to process: {len(videos_to_process)}")

    for v_idx, v_id in tqdm(enumerate(videos_to_process), total=len(videos_to_process)):
        video_path = os.path.join(base_mp4_folder, f"{v_id}.mp4")
        
        # Get total frames in video
        if config.total_video_frames is None:
            total_frames = get_total_video_frames(video_path)
        else:
            total_frames = config.total_video_frames
        
        # Get uniformly sampled frame indices
        sample_frame_indices = get_uniform_sample_frames(total_frames, num_sample_frames)
        print(f"Video {v_id}: Total frames {total_frames}, Sampling frames: {sample_frame_indices}")

        # Load video once
        video_loader = VideoLoader(video_path)

        # Process each sampled frame
        per_video_molmo_preds = {"molmo": {}}
        video_args = unique_args[v_id]
        
        for frame_idx in sample_frame_indices:
            # try:
            frame = video_loader.get_frame(frame_idx)
            # Save the frame to a temporary directory
            frame_path = save_frame_to_tmp(frame, v_id, base_tmp_dir, frame_idx)

            # Pass the saved frame path to the grounding function
            for i, caption in enumerate(video_args):
                points = point_with_molmo(model, processor, [frame_path], caption)
                if caption not in per_video_molmo_preds['molmo']:
                    per_video_molmo_preds['molmo'][caption] = {"Points": [], "Frames": []}
                if points[0] is not None:
                    per_video_molmo_preds['molmo'][caption]['Points'].append(points[0])
                    per_video_molmo_preds['molmo'][caption]['Frames'].append(frame_idx)
            # except Exception as e:
            #     print(f"Error processing frame {frame_idx} for video {v_id}: {e}")
            #     continue

        video_loader.release()
        all_video_molmo_preds[v_id] = per_video_molmo_preds

        if config.do_visualization:
            visualize_video_results(v_id, per_video_molmo_preds['molmo'], config, base_frame_dir=base_tmp_dir)

        if config.save_interval > 0:
            finished_videos.add(v_id)
                
        # except Exception as e:
        #     print(f"Error processing video {v_id}: {e}")
        #     continue

        # Save intermediate results
        if config.save_interval > 0 and (v_idx + 1) % config.save_interval == 0:
            save_results(config.output_file, all_video_molmo_preds)
            save_finished_videos(config.finished_file, finished_videos)
            print(f"Saved results for {len(all_video_molmo_preds)} videos and updated finished list.")
    
    # Save final results
    save_results(config.output_file, all_video_molmo_preds)
    if config.save_interval > 0:
        save_finished_videos(config.finished_file, finished_videos)


def process_videos_shot_molmo(config, unique_args, model, processor):
    """
    Process videos using shot-based grounding with Molmo model.

    Args:
        config (dict): Configuration settings.
        unique_args (dict): Unique arguments for grounding, keyed by video ID.
    """
    all_video_molmo_preds = {}

    base_mp4_folder = config.base_mp4_folder
    base_shot_folder = config.base_shot_folder
    base_tmp_dir = config.base_tmp_dir

    if config.follow_input_vid_list:
        # Read video IDs from the input file
        with open(config.input_vid_file, "r") as f:
            videos_to_process = sorted([line.strip() for line in f.readlines()])
    else:
        # Use the current approach to list videos in base dirs
        if config.save_interval > 0:
            finished_videos = load_finished_videos(config.finished_file)
            videos = sorted(list(unique_args.keys()))[config.video_range[0]:config.video_range[1]]
            videos_to_process = [v for v in videos if v not in finished_videos]
        else:
            videos_to_process = sorted(list(unique_args.keys()))[config.video_range[0]:config.video_range[1]]
    print(f"Total videos to process: {len(videos_to_process)}")

    # for v_idx, v_id in enumerate(videos_to_process):keep tqdm
    for v_idx, v_id in tqdm(enumerate(videos_to_process), total=len(videos_to_process)):
        try:
            video_path = os.path.join(base_mp4_folder, f"{v_id}.mp4")
            shot_file = os.path.join(base_shot_folder, v_id, v_id, f"{v_id}.videvents")

            # Read shot boundaries
            shot_boundaries = read_shot_boundaries(shot_file)

            # Load video once
            video_loader = VideoLoader(video_path)

            # Process each shot-start frame
            per_video_molmo_preds = {"molmo": {}}
            video_args = unique_args[v_id]
            
            for frame_idx in shot_boundaries:
                frame = video_loader.get_frame(frame_idx)
                try:
                # Save the frame to a temporary directory
                    frame_path = save_frame_to_tmp(frame, v_id, base_tmp_dir, frame_idx)

                    # Pass the saved frame path to the grounding function
                    for i, caption in enumerate(video_args):
                        points = point_with_molmo(model, processor, [frame_path], caption)
                        if caption not in per_video_molmo_preds['molmo']:
                            per_video_molmo_preds['molmo'][caption] = {"Points": [], "Frames": []}
                        if points[0] is not None:
                            per_video_molmo_preds['molmo'][caption]['Points'].append(points[0])
                            per_video_molmo_preds['molmo'][caption]['Frames'].append(frame_idx)
                except Exception as e:
                    print(f"Error processing frame {frame_idx} for video {v_id}: {e}")
                    continue

            video_loader.release()
            all_video_molmo_preds[v_id] = per_video_molmo_preds

            if config.do_visualization:
                visualize_video_results(v_id, per_video_molmo_preds['molmo'], config, base_frame_dir=base_tmp_dir)

            if config.save_interval > 0:
                finished_videos.add(v_id)
        except Exception as e:
            print(f"Error processing video {v_id}: {e}")

        # Save intermediate results
        if config.save_interval > 0 and (v_idx + 1) % config.save_interval == 0:
            save_results(config.output_file, all_video_molmo_preds)
            save_finished_videos(config.finished_file, finished_videos)
            print(f"Saved results for {len(all_video_molmo_preds)} videos and updated finished list.")
    save_results(config.output_file, all_video_molmo_preds)
    if config.save_interval > 0:
        save_finished_videos(config.finished_file, finished_videos)


if __name__ == "__main__":
    config_path = "config.yaml"
    config = load_config(config_path)
    unique_args = get_unique_args_for_grounding(config)
    
    if config.model_name == "molmo":
        config.output_file = config.output_file.replace("{model_name}", "molmo")
        config.finished_file = config.finished_file.replace("{model_name}", "molmo")
        model, processor = load_model_and_processor(config["molmo_model_name"], config["data_type"], device_map=config.get("device", 'auto'))
        
        if config.grounding_mode == "11_frame":
            pass # Implement if needed
        elif config.grounding_mode == "shot":
            process_videos_shot_molmo(config, unique_args, model, processor)
        elif config.grounding_mode == "uniform_sampling":
            process_videos_uniform_sampling_molmo(config, unique_args, model, processor)
        else:
            raise ValueError(f"Unsupported grounding mode: {config.grounding_mode}")
    else:
        raise ValueError(f"This script only supports molmo model, got: {config.model_name}")