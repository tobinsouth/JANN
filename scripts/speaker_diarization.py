import numpy as np
from tqdm import tqdm
from glob import glob 
import subprocess, os, json


# Load in models
from pydiar.models import BinaryKeyDiarizationModel, Segment # https://github.com/audapolis/pydiar
from pydiar.util.misc import optimize_segments
from pydub import AudioSegment
diarization_model = BinaryKeyDiarizationModel()


all_original_files = glob('../data/audiofiles/*.wav')
all_diarized_results = {}

# Loop through files and record the speaker times.
for file in tqdm(all_original_files): 

    try:
        audio = AudioSegment.from_wav(file).set_frame_rate(32000).set_channels(1)
        segments = diarization_model.diarize(32000, np.array(audio.get_array_of_samples()))
        optimized_segments = optimize_segments(segments)

        # Save the speaker times for later transcription
        audio_segments = []
        for seg in optimized_segments:
            chops = int(np.ceil(seg.length / 119))
            for i in range(chops):
                length = seg.length if chops == 1 else (119 if i < chops else seg.length - (chops - 1) * 119)
                audio_segments.append({'start': seg.start + i*119, 'len': length, 'end': seg.start + length + i*119, 'speaker_id': int(seg.speaker_id)})    

        # Update master and write to file    
        all_diarized_results.update({file: audio_segments})
        with open('../data/diarized_results.json', 'w') as f:
            json.dump(all_diarized_results, f)
    except:
        print('Error with file:', file)
        continue



