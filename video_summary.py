import cv2
import numpy as np


def visualize_video_frames(video_path, num_cols, frame_size, key_frames):
  """
  Extracts frames from a video, downsamples them, adds borders to key frames, 
  and concatenates them into a single image.

  Args:
    video_path: Path to the video file.
    num_cols: Number of columns in the output image.
    frame_size: Target size (width, height) for each downsampled frame.
    key_frames: A list of frame indices to be highlighted with a border.

  Returns:
    A numpy array representing the concatenated image.
  """

  cap = cv2.VideoCapture(video_path)
  frames = []
  
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
      frame = cv2.resize(frame, frame_size)
      frames.append(frame)
    else:
      break
  cap.release()

  # Add cyan border to key frames
  for i in key_frames:
    if 0 <= i < len(frames):
      cv2.rectangle(frames[i], (0, 0), (frame_size[0]-1, frame_size[1]-1), (255, 255, 0), 5)

  # Calculate number of rows needed
  num_rows = int(np.ceil(len(frames) / num_cols))

  # Create a blank canvas for the output image
  output_img = np.zeros((num_rows * frame_size[1], num_cols * frame_size[0], 3), dtype=np.uint8)

  # Paste the frames onto the canvas
  for i, frame in enumerate(frames):
    row = i // num_cols
    col = i % num_cols
    output_img[row * frame_size[1]: (row + 1) * frame_size[1], 
               col * frame_size[0]: (col + 1) * frame_size[0], :] = frame

  return output_img

