import librosa
import argparse
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, load_dataset

MAX_INPUT_LENGTH = 16000 * 30
model_id = "openai/whisper-small"
lang = "vietnamese"
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference of speech to text')
    parser.add_argument('--model', type=str, default='save-finetuned', help='path to audio original folder')
    parser.add_argument('--data', type=str, default='dev.wav', help='path to csv audio original folder')

    args = parser.parse_args()

    # processor = WhisperProcessor.from_pretrained("processor-pretrained")
    processor = WhisperProcessor.from_pretrained(model_id, language=lang, task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")

    # load audio sample
    sample, sr = librosa.load(args.data, sr=16000)
    sample_batch = [sample[i:i + MAX_INPUT_LENGTH] for i in range(0, len(sample), MAX_INPUT_LENGTH)]
    input_features = processor(sample_batch, sampling_rate=sr, return_tensors="pt").input_features

    # generate token ids
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print("result###", transcription[0])
    