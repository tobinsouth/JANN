# Convert all files to .wav
# from glob import glob
# import subprocess
# from tqdm import tqdm
# all_original_files = glob('../data/audiofiles/*.mp3')
# for file in tqdm(all_original_files):
#     subprocess.run(['ffmpeg', '-i',file,'-vn','-acodec','pcm_s16le','-ar','44100','-ac','2', file.replace('.mp3', '.wav'), '-y', '-loglevel','quiet'],  stdout=subprocess.DEVNULL)
#     subprocess.run(['rm', '-rf', file],  stdout=subprocess.DEVNULL)




from glob import glob
from tqdm import tqdm
import subprocess, os
from asrecognition import ASREngine

all_original_files = glob('../data/audiofiles/*.wav')
asr = ASREngine("es", model_path='jonatasgrosman/wav2vec2-large-xlsr-53-spanish', device='cuda')

# all_original_files = [f for f in all_original_files if not os.path.exists(f.replace('.wav', '.txt').replace('audiofiles', 'transcripts'))]

for file in tqdm(all_original_files): 

    subprocess.run(['ffmpeg', '-i', file, '-f', 'segment', '-segment_time', '119', '-c', 'copy', '../data/audiofiles/temp/split%03d.wav', '-loglevel','quiet'],  stdout=subprocess.DEVNULL)

    # Run speech recognition
    audio_paths = glob("../data/audiofiles/temp/*.wav")
    transcriptions = [asr.transcribe([audio_path])[0] for audio_path in tqdm(audio_paths)] # Avoids multithreading to use cuda properly

    # Save transcriptions
    tmap = {int(t['path'][-7:-4]): t['transcription'] for t in transcriptions}
    full_transcription = ' '.join(tmap[i] for i in range(len(tmap)))
    open("../data/transcripts/"+file.split('/')[-1].split('.')[0]+'.txt', 'w').write(full_transcription)

    # Remove temporary files
    subprocess.run(['rm', '-rf', '../data/audiofiles/temp/*'],  stdout=subprocess.DEVNULL)