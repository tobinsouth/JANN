from glob import glob
from tqdm import tqdm
import subprocess, os, json

thread = 3 # Choose from 4 threads

# Load in spanish speech2text model
from asrecognition import ASREngine
asr = ASREngine("es", model_path='facebook/wav2vec2-large-xlsr-53-spanish', device = 'cuda:{}'.format(thread))

# Read in json file
with open('../data/diarized_results.json') as f:
    diarized_results = json.load(f)

diarized_results = {file:speaker_sessions for file, speaker_sessions in diarized_results.items() if len(file) % 4==thread}
tempfilename = 'output{}.wav'.format(thread)

# Remove leftover files
subprocess.run(['rm', '-f', '../data/audiofiles/temp/'+tempfilename])

for file, speaker_sessions in tqdm(diarized_results.items()): 
    file_transcriptions = {}
    for i, session in enumerate(speaker_sessions):
        start, length, speaker_id = session['start'], session['len'], session['speaker_id']

        # Get segment of file
        subprocess.run(['ffmpeg','-ss',str(start),'-t',str(length),'-i',file, '../data/audiofiles/temp/'+tempfilename,'-loglevel','quiet'],   stdout=subprocess.DEVNULL)

        # Run speech recognition
        try:
            transcription = asr.transcribe(['../data/audiofiles/temp/'+tempfilename])[0]
            transcription = transcription['transcription']
        except RuntimeError:
            print('Failed on file:', file, '; Index:', i)
            transcription = ""

        transcription_info = {'i': i, 'speaker_id': speaker_id, 'length':length, 'transcription': transcription}
        file_transcriptions[i] = transcription_info

        # Save as json.
        with open('../data/transcriptions/'+file.split('/')[-1].replace('.wav','.json'), 'w') as f:
            json.dump(file_transcriptions, f)

        subprocess.run(['rm', '-f', '../data/audiofiles/temp/'+tempfilename])