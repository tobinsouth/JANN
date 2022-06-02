# https://huggingface.co/facebook/s2t-medium-mustc-multilingual-st

import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import soundfile as sf


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

batch = {'file':'../data/fullmovieaudio.wav'}
array = map_to_array(batch)["speech"]
array = array.sum(axis=1)/2

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")

inputs = processor(array, sampling_rate=16_000, return_tensors="pt")

generated_ids = model.generate(
    input_ids=inputs["input_features"],
    attention_mask=inputs["attention_mask"],
    forced_bos_token_id=processor.tokenizer.lang_code_to_id["es"]
)
translation_es = processor.batch_decode(generated_ids)



model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-covost2-es-en-st")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-covost2-es-en-st")

inputs = processor(array, sampling_rate=48_000, return_tensors="pt") # 16_000

generated_ids = model.generate(input_ids=inputs["input_features"], attention_mask=inputs["attention_mask"])
translation_en = processor.batch_decode(generated_ids, skip_special_tokens=True)



from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-spanish")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-spanish")

inputs = processor(array, sampling_rate=16_000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.batch_decode(predicted_ids)




from asrecognition import ASREngine

asr = ASREngine("es", model_path='facebook/wav2vec2-large-xlsr-53-spanish', device='cuda')
translation_asr = asr.transcribe([batch['file']])[0]['transcription']






import torchaudio
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-xlsr-53-spanish').to('cuda')
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-xlsr-53-spanish')

speech, rate = torchaudio.load(batch["file"])
speech = speech.sum(axis=0)/2
resampler = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16_000)
speechR = resampler.forward(speech.squeeze(0)).numpy()
features = processor(speechR, sampling_rate=16_000, return_tensors="pt")

input_values = features.input_values.to('cuda')
attention_mask = features.attention_mask.to('cuda')
with torch.no_grad():
    logits = model(input_values, attention_mask=attention_mask).logits
pred_ids = torch.argmax(logits, dim=-1)
torchWav_res = processor.batch_decode(pred_ids)

print(translation_es, translation_en, predicted_sentences, translation_asr, torchWav_res)


# # Testing
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
# ds = ds.map(map_to_array)

# model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
# processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")

# inputs = processor(torch.as_tensor(ds["speech"][0]), sampling_rate=16_000, return_tensors="pt")

# generated_ids = model.generate(
#     input_ids=inputs["input_features"],
#     attention_mask=inputs["attention_mask"],
#     forced_bos_token_id=processor.tokenizer.lang_code_to_id["en"]
# )
# translation_es =processor.batch_decode(generated_ids)



# model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-covost2-es-en-st")
# processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-covost2-es-en-st")

# inputs = processor(torch.as_tensor(ds["speech"][0]), sampling_rate=48_000, return_tensors="pt") # 16_000

# generated_ids = model.generate(input_ids=inputs["input_features"], attention_mask=inputs["attention_mask"])
# translation_en =  processor.batch_decode(generated_ids, skip_special_tokens=True)




# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-spanish")
# model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-spanish")

# inputs = processor(ds["speech"][0], sampling_rate=16_000, return_tensors="pt", padding=True)
# with torch.no_grad():
#     logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

# predicted_ids = torch.argmax(logits, dim=-1)
# predicted_sentences = processor.batch_decode(predicted_ids)






# New module
# pip install SpeechRecognition
import speech_recognition as sr


r = sr.Recognizer()
with sr.AudioFile('../data/output.wav') as source:
    audio = r.record(source)  # read the entire audio file

r.recognize_sphinx(audio, language='es-ES', show_all=True)


from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *

# Here is the configuration for Spanish
config = Decoder.default_config()
config.set_string('-hmm', 'cmusphinx-es-5.2/model_parameters/voxforge_es_sphinx.cd_ptm_4000')
config.set_string('-lm', 'es-20k.lm.gz')
config.set_string('-dict', 'es.dict')
decoder = Decoder(config)


decoder = Decoder(config)
decoder.start_utt()
stream = open('hola.wav', 'rb')