from datasets import load_dataset, DatasetDict
from transformers import Seq2SeqTrainer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from datasets import Audio
import torch
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
import time
import os
import shutil

from datetime import datetime

# Get current date and time
current_time = datetime.now()

# Format the timestamp to include year, month, date, hour, and minute
timestamp = current_time.strftime("%Y-%m-%d-%H-%M")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# feature_extractor = WhisperFeatureExtractor.from_pretrained(r"feature_extractor", local_files_only=True)
# tokenizer = WhisperTokenizer.from_pretrained(r"tokenizer_pretrained", language="vietnamese", task="transcribe")
# processor = WhisperProcessor.from_pretrained(r"processor_pretrained", language="vietnamese", task="transcribe")

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="vietnamese", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="vietnamese", task="transcribe")


metric = evaluate.load("wer")

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # print ("*******************************")
    # print (batch)
    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def trainingApi(excel_dataset='./dataset/final_script.csv', 
                per_device_train_batch_size=16, 
                learning_rate=1e-5, 
                warmup_steps=500, 
                max_steps=4000, 
                per_device_eval_batch_size=8, 
                generation_max_length=225, 
                save_steps=1000, 
                eval_steps=1000, 
                logging_steps=25, 
                txtfile='txtfile', 
                **kwargs):
   
# def trainingApi(excel_dataset='./manh_data/vi.csv', per_device_train_batch_size=16, learning_rate=1e-5, warmup_steps=500, max_steps=4000, per_device_eval_batch_size=8, generation_max_length=225, save_steps=1000, eval_steps=1000, logging_steps=25, txtfile='txtfile', **kwargs):
   
    common_voice = DatasetDict()

    # output_dir = str(time.time())

    output_dir = f"training/{timestamp}"
    # common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation", use_auth_token=True)
    # # common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test", use_auth_token=True)

    # common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    # print(common_voice["train"])


    datatest = DatasetDict()

    # datatest = load_dataset('csv', data_files=['./dataset/final_script.csv'],split="train")
    datatest = load_dataset('csv', data_files=[excel_dataset],split="train")

    # datatest['test'] = load_dataset('csv', data_files=['dataset/final_script.csv'])
    common_voice = datatest.train_test_split(shuffle = True, seed = 200, test_size=0.3)
    # datatest

    print(common_voice)
    # exit()


    # feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    # tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="vietnamese", task="transcribe")
    # processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="vietnamese", task="transcribe")

    # tokenizer.save_pretrained('tokenizer-pretrained')
    # processor.save_pretrained('processor-pretrained')
    # feature_extractor.save_pretrained('feature_extractor')

    print(common_voice["train"][0])

    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    # model.save_pretrained('./pretrained')
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []


    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,  # change to a repo name of your choice
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=generation_max_length,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        # report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)

    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # print("model###", output_dir)
    trainer.train()
    best_model = f'best_{timestamp}'
    best_model = os.path.join('model', best_model)
    trainer.save_model(best_model)
    # trainer.sa
    f=open(txtfile,"w")
    f.write('model###'+best_model)
    f.close()
    shutil.rmtree(output_dir)
    return txtfile

    # exit()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' Parser for flow training speech to text')
    # parser.add_argument('--excel_dataset', type=str, default='./dataset/final_script.csv', help='path to annotation file')
    parser.add_argument('--excel_dataset', type=str, default='./manh_data/vi.csv', help='path to annotation file')
    parser.add_argument('--per_device_train_batch_size', type=int, default=16, help='train batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--warmup_steps', type=int, default=5, help='warmup steps')
    parser.add_argument('--max_steps', type=int, default=40, help='max steps')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help='max steps')
    parser.add_argument('--generation_max_length', type=int, default=225, help='max length generate')
    parser.add_argument('--save_steps', type=int, default=10, help='save step')
    parser.add_argument('--eval_steps', type=int, default=10, help='eval step')
    parser.add_argument('--logging_steps', type=int, default=2, help='logging steps')
    parser.add_argument('--txtfile', type=str, default='txtfile.txt', help='output file')

    args = vars(parser.parse_args())
    print("args", args)
    print("txtfile123", args['txtfile'])


    best_model = trainingApi(**args)

    # print("txtfile123", args.txtfile)

    