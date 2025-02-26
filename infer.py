import librosa

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, load_dataset

MAX_INPUT_LENGTH = 16000 * 30

# load model and processor
processor = WhisperProcessor.from_pretrained("processor-pretrained")
model = WhisperForConditionalGeneration.from_pretrained(r"model\best_1695629310.9396894")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="vietnamese", task="transcribe")

# load audio sample
sample, sr = librosa.load("http://10.0.68.103:8889/denoise_file?path=/home/ai-ubuntu/hddnew/HungDangT04/API_speech_enhancement/data_folder/save_predictions/noisereduce_lib/test1_denoise.wav", sr=16000)
sample_batch = [sample[i:i + MAX_INPUT_LENGTH] for i in range(0, len(sample), MAX_INPUT_LENGTH)]
input_features = processor(sample_batch, sampling_rate=sr, return_tensors="pt").input_features

# generate token ids
predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)