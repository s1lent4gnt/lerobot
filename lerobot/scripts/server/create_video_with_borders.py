#!/usr/bin/env python3
"""
Script to generate a video from frames with colored borders for frames with intervention.
"""

import os
import cv2
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

def draw_rounded_rectangle(image, top_left, bottom_right, color, thickness=-1, radius=10):
    """
    Draw a rounded rectangle on an image.
    
    Args:
        image: The image to draw on
        top_left: The top-left corner coordinates (x, y)
        bottom_right: The bottom-right corner coordinates (x, y)
        color: The color of the rectangle (BGR)
        thickness: The thickness of the rectangle border, -1 for filled rectangle
        radius: The radius of the rounded corners
        
    Returns:
        The image with the rounded rectangle drawn on it
    """
    # Unpack coordinates
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Ensure radius doesn't exceed half the width or height
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    
    # If thickness is negative (filled), draw a filled rectangle and then overlay rounded corners
    if thickness < 0:
        # Draw the main rectangle
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        
        # Draw the four corner circles with anti-aliasing
        cv2.circle(image, (x1 + radius, y1 + radius), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(image, (x2 - radius, y1 + radius), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(image, (x1 + radius, y2 - radius), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(image, (x2 - radius, y2 - radius), radius, color, thickness, cv2.LINE_AA)
    else:
        # For non-filled rectangles, draw the four sides
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw the four corner arcs with anti-aliasing
        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
    
    return image

def add_text_with_box(image, text, position, font, font_scale, text_color, box_color, thickness, opacity, config=None):
    """
    Add text with a colored box background with rounded corners to an image with a specific opacity.
    
    Args:
        image: The image to add text to
        text: The text to add
        position: The position (x, y) to place the text
        font: The font to use
        font_scale: The font scale
        text_color: The text color (BGR)
        box_color: The box background color (BGR)
        thickness: The text thickness
        opacity: The opacity of the text and box (0.0 to 1.0)
        config: Configuration dictionary with box settings (optional)
    
    Returns:
        The image with the text and box added
    """
    # Create a copy of the image
    overlay = image.copy()
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate box coordinates
    box_padding = 10  # Padding around text
    box_x1 = position[0] - box_padding
    box_y1 = position[1] - text_height - box_padding
    box_x2 = position[0] + text_width + box_padding
    box_y2 = position[1] + box_padding
    
    # Determine the corner radius
    corner_radius = 10  # Default radius
    if config is not None and 'box' in config and 'corner_radius' in config['box']:
        corner_radius = config['box']['corner_radius']
    
    # Draw the rounded box on the overlay
    draw_rounded_rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), box_color, -1, radius=corner_radius)
    
    # Add the text to the overlay with anti-aliasing
    cv2.putText(overlay, text, position, font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    # Blend the overlay with the original image based on opacity
    output = cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)
    
    return output

def format_time(frame_number, fps):
    """
    Format the time based on frame number and fps.
    
    Args:
        frame_number: The current frame number (0-based)
        fps: Frames per second
        
    Returns:
        A formatted time string (HH:MM:SS.mmm)
    """
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def add_timestamp_label(frame, frame_number, fps, width, config):
    """
    Add a timestamp label to the frame.
    
    Args:
        frame: The frame to add the label to
        frame_number: The current frame number
        fps: Frames per second
        width: The width of the frame
        config: Configuration dictionary with font and opacity settings
        
    Returns:
        The frame with the timestamp label added
    """
    # Format the time based on frame index and fps
    time_text = format_time(frame_number, fps)
    
    # Calculate position for time label in top-right corner
    # First get the text size to position it properly
    (time_text_width, time_text_height), _ = cv2.getTextSize(
        time_text, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        config['font']['scale'], 
        config['font']['thickness']
    )
    time_position = (width - time_text_width - 30, 50)  # 30 pixels padding from right edge
    
    # Add time label with a black box background
    return add_text_with_box(
        frame,
        time_text,
        time_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        config['font']['scale'],
        (255, 255, 255),  # White text
        (0, 0, 0),        # Black box
        config['font']['thickness'],
        config['opacity']['high'],  # Always high opacity for time
        config
    )

def add_rl_policy_label(frame, is_intervention, config):
    """
    Add an RL Policy label to the frame.
    
    Args:
        frame: The frame to add the label to
        is_intervention: Whether this frame has intervention
        config: Configuration dictionary with font and opacity settings
        
    Returns:
        The frame with the RL Policy label added
    """
    # Set opacity based on intervention status (active when no intervention)
    rl_opacity = config['opacity']['high'] if not is_intervention else config['opacity']['low']
    
    # Position for RL Policy label
    rl_position = (30, 50)  # Top-left corner
    
    # Colors
    white_text_color = (255, 255, 255)  # White text
    green_box_color = (0, 255, 0)       # Green box for active RL Policy
    grey_box_color = (128, 128, 128)    # Grey box for inactive RL Policy
    
    # Determine box color based on intervention status
    rl_box_color = green_box_color if not is_intervention else grey_box_color
    
    # Add RL Policy text with box and appropriate opacity
    return add_text_with_box(
        frame, 
        "RL Policy", 
        rl_position, 
        config['font']['type'], 
        config['font']['scale'], 
        white_text_color,  # White text
        rl_box_color,      # Box color based on status
        config['font']['thickness'], 
        rl_opacity,
        config
    )

def add_intervention_label(frame, is_intervention, config):
    """
    Add an Expert Intervention label to the frame.
    
    Args:
        frame: The frame to add the label to
        is_intervention: Whether this frame has intervention
        config: Configuration dictionary with font and opacity settings
        
    Returns:
        The frame with the Expert Intervention label added
    """
    # Set opacity based on intervention status (active when intervention)
    intervention_opacity = config['opacity']['high'] if is_intervention else config['opacity']['low']
    
    # Position for Expert Intervention label
    intervention_position = (30, 100)  # Below RL Policy text
    
    # Colors
    white_text_color = (255, 255, 255)  # White text
    red_box_color = (0, 0, 255)         # Red box for active Expert Intervention
    grey_box_color = (128, 128, 128)    # Grey box for inactive Expert Intervention
    
    # Determine box color based on intervention status
    intervention_box_color = red_box_color if is_intervention else grey_box_color
    
    # Add Expert Intervention text with box and appropriate opacity
    return add_text_with_box(
        frame, 
        "Expert Intervention", 
        intervention_position, 
        config['font']['type'], 
        config['font']['scale'], 
        white_text_color,       # White text
        intervention_box_color, # Box color based on status
        config['font']['thickness'], 
        intervention_opacity,
        config
    )

def main():
    # Parameters
    frames_dir = "output_frames/try_3/video_frames"
    labels_file = "output_frames/try_3/intervention_labels.csv"
    output_video = "output_frames/output_video.mp4"
    intervention_border_color = (36, 54, 212)  # Red color in BGR format for intervention
    no_intervention_border_color = (71, 196, 39)  # Green color in BGR format for no intervention
    border_thickness = 10
    fps = 100  # Frames per second for output video (10 frame = 1 second)
    real_fps = 10
    
    # Global configuration dictionary
    config = {
        'font': {
            'type': cv2.FONT_HERSHEY_TRIPLEX,  # More modern sans-serif font similar to Helvetica
            'scale': 0.8,
            'thickness': 1
        },
        'opacity': {
            'low': 0.3,  # 30% opacity for inactive label
            'high': 1.0  # 100% opacity for active label
        },
        'box': {
            'corner_radius': 5  # Radius for rounded corners of label boxes
        }
    }
    
    # Read the labels
    print("Reading intervention labels...")
    labels_df = pd.read_csv(labels_file)
    
    # Create a dictionary for quick lookup of frame labels
    labels_dict = dict(zip(labels_df['frame'], labels_df['label']))
    
    # Get all frame files and sort them
    print("Finding frame files...")
    frame_files = sorted(glob(os.path.join(frames_dir, "*.png")))
    
    if not frame_files:
        print(f"No frame files found in {frames_dir}")
        return
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    
    # Initialize video writer
    print(f"Initializing video writer with dimensions {width}x{height}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Process each frame
    print("Processing frames and adding borders where needed...")
    for frame_index, frame_path in enumerate(tqdm(frame_files[:-370])):
        # Get the frame filename
        frame_filename = os.path.basename(frame_path)
        
        # Read the frame
        frame = cv2.imread(frame_path)

        # Determine if this frame has intervention (label = 1)
        has_intervention = frame_filename in labels_dict and labels_dict[frame_filename] == 1
        
        # Add border with appropriate color based on intervention status
        # Add a border overlay to the frame with anti-aliasing
        # Draw a rectangle on the frame instead of adding a border
        # This keeps the original frame size and content intact
        border_color = intervention_border_color if has_intervention else no_intervention_border_color
        cv2.rectangle(
            frame,
            (0, 0),                          # Top-left corner
            (width - 1, height - 1),         # Bottom-right corner
            border_color,                    # Border color based on intervention status
            border_thickness,                # Border thickness
            cv2.LINE_AA                      # Anti-aliasing for smoother lines
        )
        
        # Add RL Policy label
        frame = add_rl_policy_label(
            frame,
            has_intervention,
            config
        )
        
        # Add Expert Intervention label
        frame = add_intervention_label(
            frame,
            has_intervention,
            config
        )
        
        # Add timestamp label
        frame = add_timestamp_label(
            frame,
            frame_index,
            real_fps,
            width,
            config
        )
        
        # Write the frame to the video
        video_writer.write(frame)
    
    # Release the video writer
    video_writer.release()
    
    print(f"Video generation complete. Output saved to {output_video}")

if __name__ == "__main__":
    main()
