# Change the current directory to /content/
%cd /content/

# Clone the av_hubert repository from GitHub
!git clone https://github.com/facebookresearch/av_hubert.git

# Change the current directory to av_hubert/
%cd av_hubert

# Initialize and update the git submodules
!git submodule init
!git submodule update

# Install required Python packages
!pip install scipy
!pip install sentencepiece
!pip install python_speech_features
!pip install scikit-video

# Change the current directory to fairseq/
%cd fairseq

# Install the fairseq package
!pip install ./


# Create the directory /content/data/misc/ if it does not exist
!mkdir -p /content/data/misc/

# Download the shape predictor file from dlib.net and save it as shape_predictor_68_face_landmarks.dat.bz2 in the directory /content/data/misc/
!wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O /content/data/misc/shape_predictor_68_face_landmarks.dat.bz2

# Extract the shape predictor file by decompressing the file shape_predictor_68_face_landmarks.dat.bz2 using the bzip2 command
!bzip2 -d /content/data/misc/shape_predictor_68_face_landmarks.dat.bz2

# Download the 20words_mean_face.npy file from a GitHub repository and save it as 20words_mean_face.npy in the directory /content/data/misc/
!wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy -O /content/data/misc/20words_mean_face.npy

# Download the video file from a URL and save it as clip.mp4 in the directory /content/data/
# Thank you for your help
!wget --content-disposition "https://drive.google.com/file/d/1LfO2w-hsRX79_d7-7-eWU1mOcDo-Tc-s/view?usp=share_link" -O /content/data/clip.mp4

# Download the video file from a URL and save it as clip.mp4 in the directory /content/data/
#Hello how are you!
!wget --content-disposition "https://drive.google.com/uc?export=download&id=1I1t4meGxG6Z1AhkmYu5WMeycwaljHagW" -O /content/data/clip.mp4

# Change directory to the avhubert folder
%cd /content/av_hubert/avhubert/

# Import necessary libraries
import dlib, cv2, os
import numpy as np
import skvideo
import skvideo.io
from tqdm import tqdm
from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
from IPython.display import HTML
from base64 import b64encode

# Define a function to play a video in the Jupyter Notebook
def play_video(video_path, width=200):
  mp4 = open(video_path,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  return HTML(f"""
  <video width={width} controls>
        <source src="{data_url}" type="video/mp4">
  </video>
  """)

# Define a function to detect facial landmarks in an image
def detect_landmark(image, detector, predictor):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Detect faces in the grayscale image
    rects = detector(gray, 1)
    coords = None
    # For each detected face, predict the facial landmarks and save their coordinates
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Define a function to preprocess a video, including detecting facial landmarks, interpolating missing landmarks, cropping mouth regions, and saving the result as a new video
def preprocess_video(input_video_path, output_video_path, face_predictor_path, mean_face_path):
  # Load the face detector and facial landmark predictor models
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(face_predictor_path)
  # Define the size of the input frames after resizing
  STD_SIZE = (256, 256)
  # Load the mean face landmarks and stable points IDs for cropping mouth regions
  mean_face_landmarks = np.load(mean_face_path)
  stablePntsIDs = [33, 36, 39, 42, 45]
  # Read in the input video frames
  videogen = skvideo.io.vread(input_video_path)
  frames = np.array([frame for frame in videogen])
  # Detect facial landmarks in each frame
  landmarks = []
  for frame in tqdm(frames):
      landmark = detect_landmark(frame, detector, predictor)
      landmarks.append(landmark)
  # Interpolate missing facial landmarks
  preprocessed_landmarks = landmarks_interpolate(landmarks)
  # Crop mouth regions from the preprocessed landmarks and save as separate frames
  rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                        window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
  # Write the cropped frames as a new video
  write_video_ffmpeg(rois, output_video_path, "/usr/bin/ffmpeg")
  return
face_predictor_path = "/content/data/misc/shape_predictor_68_face_landmarks.dat"
mean_face_path = "/content/data/misc/20words_mean_face.npy"
origin_clip_path = "/content/data/clip.mp4"
mouth_roi_path = "/content/data/roi.mp4"
preprocess_video(origin_clip_path, mouth_roi_path, face_predictor_path, mean_face_path)
play_video(mouth_roi_path)

!pwd
%mkdir -p /content/data/
!wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/base_vox_433h.pt -O /content/data/finetune-model.pt


%cd /content/av_hubert/avhubert
import cv2
import tempfile
from argparse import Namespace
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig
from IPython.display import HTML

def predict(video_path, ckpt_path, user_dir):
  num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
  data_dir = tempfile.mkdtemp()
  tsv_cont = ["/\n", f"test-0\t{video_path}\t{None}\t{num_frames}\t{int(16_000*num_frames/25)}\n"]
  label_cont = ["DUMMY\n"]
  with open(f"{data_dir}/test.tsv", "w") as fo:
    fo.write("".join(tsv_cont))
  with open(f"{data_dir}/test.wrd", "w") as fo:
    fo.write("".join(label_cont))
  utils.import_user_module(Namespace(user_dir=user_dir))
  modalities = ["video"]
  gen_subset = "test"
  gen_cfg = GenerationConfig(beam=20)
  models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
  models = [model.eval() for model in models]
  saved_cfg.task.modalities = modalities
  saved_cfg.task.data = data_dir
  saved_cfg.task.label_dir = data_dir
  task = tasks.setup_task(saved_cfg.task)
  task.load_dataset(gen_subset, task_cfg=saved_cfg.task)
  generator = task.build_generator(models, gen_cfg)

  def decode_fn(x):
      dictionary = task.target_dictionary
      symbols_ignore = generator.symbols_to_strip_from_output
      symbols_ignore.add(dictionary.pad())
      return task.datasets[gen_subset].label_processors[0].decode(x, symbols_ignore)

  itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(shuffle=False)
  sample = next(itr)
  # sample = utils.move_to_cuda(sample)
  hypos = task.inference_step(generator, models, sample)
  ref = decode_fn(sample['target'][0].int().cpu())
  hypo = hypos[0][0]['tokens'].int().cpu()
  hypo = decode_fn(hypo)
  return hypo

mouth_roi_path, ckpt_path = "/content/data/roi.mp4", "/content/data/finetune-model.pt"
user_dir = "/content/av_hubert/avhubert"
hypo = predict(mouth_roi_path, ckpt_path, user_dir)
HTML(f"""
  <h3>
    Prediction - {hypo}
  </h3>
  """)
