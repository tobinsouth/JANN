# Convert all files to .wav
from glob import glob
import subprocess
from tqdm import tqdm
all_original_files = glob('../data/audiofiles/*.mp3')
for file in tqdm(all_original_files):
    subprocess.run(['ffmpeg', '-i',file,'-vn','-acodec','pcm_s16le','-ar','44100','-ac','2', file.replace('.mp3', '.wav'), '-y', '-loglevel','quiet'],  stdout=subprocess.DEVNULL)
    subprocess.run(['rm', '-rf', file],  stdout=subprocess.DEVNULL)
