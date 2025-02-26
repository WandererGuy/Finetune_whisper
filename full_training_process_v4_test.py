import shutil, os, random
from sklearn.utils import shuffle
import logging
# from step1_prepare_dataset import get_transcription, create_json
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
# from gen_txt import json_to_text
# from step3_train_lm import dataio_prepare, LM
# from step4_train_speech_reconizer import ASR, dataio_prepare_step4
from speechbrain.utils.distributed import run_on_main
import argparse
import glob
import time
import os
import signal
import shutil
import torch
from flask import Flask, request, jsonify,Response 
# import pyodbc
# conn = pyodbc.connect('Driver={SQL Server};'
#                     'Server=10.0.68.100;'
#                     'Database=testdb_video1;'
#                     'UID=sa;'
#                     'PWD=trinamA@123ai;'
#                     , autocommit=True)
# cursor = conn.cursor() 

import datetime






#Define variable:
# logger = logging.getLogger(__name__)
# SAMPLERATE = 16000
# import pandas as pd
# import gc
# def step0_gen_dataset_function(input_path,excelpath):
#     # input_path = 'dataset/audio'
#         output_path = 'dataset_processing/step0_gen_dataset'
#     # try:
#         if os.path.exists(os.path.join(output_path, 'train')):
#             shutil.rmtree(os.path.join(output_path, 'train'))
#         if os.path.exists(os.path.join(output_path, 'test')):
#             shutil.rmtree(os.path.join(output_path, 'test'))
#         if os.path.exists(os.path.join(output_path, 'valid')):
#             shutil.rmtree(os.path.join(output_path, 'valid'))
            
#         # os.makedirs(output_path)
#         if not os.path.exists(os.path.join(output_path, 'train')):
#             os.makedirs(os.path.join(output_path, 'train'))
#         if not os.path.exists(os.path.join(output_path, 'test')):
#             os.makedirs(os.path.join(output_path, 'test'))
#         if not os.path.exists(os.path.join(output_path, 'valid')):
#             os.makedirs(os.path.join(output_path, 'valid'))

#         df = pd.read_csv(excelpath, encoding='utf-8')
#         listAudio=[i +".wav" for i in df['id']]

#         # random_filename = shuffle(os.listdir(input_path), random_state=0)
#         random_filename = shuffle(listAudio, random_state=0)
     
#         test_count = int((len(random_filename) / 100) * 10)
#         for t in range(test_count):
#             choiced_test_name = random_filename[0]
#             if not os.path.exists(os.path.join(output_path, 'valid', choiced_test_name)):
#                 shutil.copy(os.path.join(input_path, choiced_test_name), os.path.join(output_path, 'test'))
#             random_filename.pop(0)

#         valid_count = int((len(random_filename) / 100) * 20)
#         for v in range(valid_count):
#             choiced_valid_name = random_filename[v]
#             if not os.path.exists(os.path.join(output_path, 'valid', choiced_valid_name)):
#                 shutil.copy(os.path.join(input_path, choiced_valid_name), os.path.join(output_path, 'valid'))
#             random_filename.pop(0)

#         print("new random filename: ", random_filename)
#         for file in random_filename:
#             shutil.copy(os.path.join(input_path, file), os.path.join(output_path, 'train'))
#         result_gen_dataset = 1
#     # except:
#     #     result_gen_dataset = 0

#         return result_gen_dataset


# def step1_prepare_dataset_function(label_file):

#     # try:
#         output_step1 = 'dataset_processing/step1_json_output'
#         if os.path.exists(output_step1):
#             shutil.rmtree(output_step1)
#         os.makedirs(output_step1)

#         # trans_dict = get_transcription("dataset/excel/final_script.csv")
#         trans_dict = get_transcription(label_file)
#         print(trans_dict)
#         create_json("dataset_processing/step0_gen_dataset/test", trans_dict,
#                     "dataset_processing/step1_json_output/test.json")
#         create_json("dataset_processing/step0_gen_dataset/train", trans_dict,
#                     "dataset_processing/step1_json_output/train.json")
#         create_json("dataset_processing/step0_gen_dataset/valid", trans_dict,
#                     "dataset_processing/step1_json_output/valid.json")
        
#         result_prepare_dataset = 1
#     # except Exception as e:
#     #     print(e)
#     #     result_prepare_dataset = 0
#         del trans_dict
#         gc.collect()
#         torch.cuda.empty_cache()
#         return result_prepare_dataset

# def step2_train_tokenizer_function(override_step2=''):
#     try:
#         output_step2 = 'dataset_processing/step2_tokenizer'
#         if os.path.exists(output_step2):
#             shutil.rmtree(output_step2)
#         os.makedirs(output_step2)
#         # Load hyperparameters file with command-line overrides
#         # hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
#         hparams_file = 'step2_tokenizer.yaml'
#         run_opts = {'debug': False, 'debug_batches': 2, 'debug_epochs': 2, 'device': 'cuda:0',
#                     'data_parallel_backend': False, 'distributed_launch': False, 'distributed_backend': 'nccl',
#                     'find_unused_parameters': False}
#         overrides = override_step2
#         with open(hparams_file) as fin:
#             hparams = load_hyperpyyaml(fin, overrides)

#         # Create experiment directory
#         sb.create_experiment_directory(
#             experiment_directory=hparams["output_folder"],
#             hyperparams_to_save=hparams_file,
#             overrides=overrides,
#         )

#         # Train tokenizer
#         hparams["tokenizer"]()

#         result_train_tokenizer = 1
#         del hparams
#         gc.collect()
#         torch.cuda.empty_cache()
#     except Exception as e:
#         print(e)
#         result_train_tokenizer = 0
#     return result_train_tokenizer

# def step3_train_language_model_function(override_step3=''):
#     # Gen text file of valid and test
#     json_to_text("dataset_processing/step1_json_output/test.json", "dataset_processing/step1_json_output/test.txt")
#     json_to_text("dataset_processing/step1_json_output/valid.json", "dataset_processing/step1_json_output/valid.txt")

#     output_step3 = 'dataset_processing/step3_language_model'
#     try:
#         if os.path.exists(output_step3):
#             shutil.rmtree(output_step3)
#         os.makedirs(output_step3)

#         # Reading command line arguments
#         #hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
#         hparams_file = 'step3_RNNLM.yaml'
#         run_opts = {'debug': False, 'debug_batches': 2, 'debug_epochs': 2, 'device': 'cuda:0',
#                    'data_parallel_backend': False, 'distributed_launch': False, 'distributed_backend': 'nccl',
#                    'find_unused_parameters': False}
#         overrides = override_step3

#         # Initialize ddp (useful only for multi-GPU DDP training)
#         sb.utils.distributed.ddp_init_group(run_opts)

#         # Load hyperparameters file with command-line overrides
#         with open(hparams_file) as fin:
#             hparams = load_hyperpyyaml(fin, overrides)

#         # Create experiment directory
#         sb.create_experiment_directory(
#             experiment_directory=hparams["output_folder"],
#             hyperparams_to_save=hparams_file,
#             overrides=overrides,
#         )

#         # Create dataset objects "train", "valid", and "test"
#         train_data, valid_data, test_data = dataio_prepare(hparams)

#         # Initialize the Brain object to prepare for LM training.
#         lm_brain = LM(
#             modules=hparams["modules"],
#             opt_class=hparams["optimizer"],
#             hparams=hparams,
#             run_opts=run_opts,
#             checkpointer=hparams["checkpointer"],
#         )

#         # The `fit()` method iterates the training loop, calling the methods
#         # necessary to update the parameters of the model. Since all objects
#         # with changing state are managed by the Checkpointer, training can be
#         # stopped at any point, and will be resumed on next call.
#         lm_brain.fit(
#             lm_brain.hparams.epoch_counter,
#             train_data,
#             valid_data,
#             train_loader_kwargs=hparams["train_dataloader_opts"],
#             valid_loader_kwargs=hparams["valid_dataloader_opts"],
#         )

#         # Load best checkpoint for evaluation
#         test_stats = lm_brain.evaluate(
#             test_data,
#             min_key="loss",
#             test_loader_kwargs=hparams["test_dataloader_opts"],
#         )
#         result_train_lm = 1
#         del train_data
#         del valid_data
#         del test_data
#         del lm_brain
#         del test_stats
#         gc.collect()
#         torch.cuda.empty_cache()
#     except Exception as e:
#         print(e)
#         result_train_lm = 0
#     return result_train_lm

# def step4_train_speech_recognizer_function(override_step4='batch_size: 1'):
#     output_step4 = 'dataset_processing/step4_train_speech_recognizer'
#     try:
#         if os.path.exists(output_step4):
#             shutil.rmtree(output_step4)
#         os.makedirs(output_step4)

#         # Reading command line arguments
#         # hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
#         hparams_file = 'step4.yaml'
#         run_opts = {'debug': False, 'debug_batches': 2, 'debug_epochs': 2, 'device': 'cuda:0',
#                     'data_parallel_backend': False, 'distributed_launch': False, 'distributed_backend': 'nccl',
#                     'find_unused_parameters': False}
#         overrides = override_step4

#         # Initialize ddp (useful only for multi-GPU DDP training)
#         sb.utils.distributed.ddp_init_group(run_opts)

#         # Load hyperparameters file with command-line overrides
#         with open(hparams_file) as fin:
#             hparams = load_hyperpyyaml(fin, overrides)

#         # Create experiment directory
#         sb.create_experiment_directory(
#             experiment_directory=hparams["output_folder"],
#             hyperparams_to_save=hparams_file,
#             overrides=overrides,
#         )

#         # We can now directly create the datasets for training, valid, and test
#         datasets = dataio_prepare_step4(hparams)

#         # In this case, pre-training is essential because mini-librispeech is not
#         # big enough to train an end-to-end model from scratch. With bigger dataset
#         # you can train from scratch and avoid this step.
#         # We download the pretrained LM from HuggingFace (or elsewhere depending on
#         # the path given in the YAML file). The tokenizer is loaded at the same time.
#         run_on_main(hparams["pretrainer"].collect_files)
#         hparams["pretrainer"].load_collected(device=run_opts["device"])
#         # Trainer initialization
#         asr_brain = ASR(
#             modules=hparams["modules"],
#             opt_class=hparams["opt_class"],
#             hparams=hparams,
#             run_opts=run_opts,
#             checkpointer=hparams["checkpointer"],
#         )

#         # The `fit()` method iterates the training loop, calling the methods
#         # necessary to update the parameters of the model. Since all objects
#         # with changing state are managed by the Checkpointer, training can be
#         # stopped at any point, and will be resumed on next call.
       
#         asr_brain.fit(
#             asr_brain.hparams.epoch_counter,
#             datasets["train"],
#             datasets["valid"],
#             train_loader_kwargs=hparams["train_dataloader_opts"],
#             valid_loader_kwargs=hparams["valid_dataloader_opts"],
#         )
       
#         # Load best checkpoint for evaluation
#         # test_stats = asr_brain.evaluate(
#         #     test_set=datasets["test"],
#         #     min_key="WER",
#         #     test_loader_kwargs=hparams["test_dataloader_opts"],
#         # )
#         del hparams
#         del asr_brain
#         del datasets
#         gc.collect()
#         torch.cuda.empty_cache()
#         return_train_sr = 1
#     except Exception as e:
#         print(e)
#         return_train_sr = 0
#     return return_train_sr
from werkzeug.utils import secure_filename
import datetime as dt
# # def get_newest(path):
#     dir_path = path

#     # create a list of all the directories in the specified directory
#     all_dirs = glob.glob(os.path.join(dir_path, '*/'))

#     # sort the list of directories by creation time in descending order
#     all_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)

#     # get the newest folder by selecting the first element of the sorted list
#     if len(all_dirs)==0:
#         return "ERROR"
#     newest_folder = all_dirs[0]

#     return newest_folder

from speechbrain.pretrained import EncoderDecoderASR, EncoderASR
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
# from flask import send_from_directory

# @app.route('/predict/<path:path>',methods=['GET'])
# def send_report(path):
#     return send_from_directory('predict', path)

@app.route('/',methods=['GET'])
def index1():
    return "Done"

# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     start_time = time.time()
#     if request.method == 'POST':
#         test_file = request.files['file']
     
#         filename = test_file.filename
#         print(filename)
#         if filename.endswith('.wav'):
#             time_now = dt.datetime.now()
#             # time_now=dt.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
#             time_now=dt.datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-4]
#             new_file_name =secure_filename(test_file.filename)

#             file_path = os.path.join('static', new_file_name)
     
#             test_file.save(file_path)
                
                
#             return jsonify({"code":200,"file_path":os.path.join(r"C:\HungDangT04\speechbrain_stt_training",file_path),'message': "http://10.0.68.100:5019/"+file_path})
#         else:
#             return jsonify({"code":500,'file_path':"",'message': 'Failed, try to upload another file'})
#     return jsonify({'message': 'Error Occured'})

# @app.route('/infer',methods=['POST'])
# def inferApi():
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     model = request.form.get('model','dataset/audio')
#     data = request.form.get('data','dataset/excel')
#     name_res = model
#     asrpath = os.path.join(name_res, 'asr.ckpt')
#     lmpath = os.path.join(name_res, 'lm.ckpt')
#     tk_path = os.path.join(name_res, 'tokenizer.model')
#     hparams_file = 'pretrained/hyperparams.yaml'
#     overrides = {'asrpath': asrpath, 'lmpath': lmpath, 'tokenizerpath': tk_path}
#     with open(hparams_file) as fin:
#         hparams = load_hyperpyyaml(fin, overrides)
#     asr_model = EncoderDecoderASR.from_hparams(source=model, savedir=model)
#     print("data",data)
#     res = asr_model.transcribe_file(path=data)
#     del asr_model
#     gc.collect()
#     torch.cuda.empty_cache()
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#     print('result###', res)
#     return 'result###'+res
# @app.route('/inferfile',methods=['POST'])
# def inferfile():
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     model = request.form.get('model','dataset/audio')
#     data = request.form.get('data','dataset/excel')
#     if "http://10.0.68.100" in data:
#         data=data
#     else:
#         test_file = request.files['file']
#         filename = test_file.filename
#         if filename.endswith('.wav'):
#             time_now = dt.datetime.now()
#             # time_now=dt.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
#             time_now=dt.datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-4]
#             new_file_name = str(time_now)+'_'+secure_filename(test_file.filename)
#             file_path = os.path.join('predict', new_file_name)
#             test_file.save(file_path)
#     data=file_path
#     name_res = model
#     asrpath = os.path.join(name_res, 'asr.ckpt')
#     lmpath = os.path.join(name_res, 'lm.ckpt')
#     tk_path = os.path.join(name_res, 'tokenizer.model')
#     hparams_file = 'pretrained/hyperparams.yaml'
#     overrides = {'asrpath': asrpath, 'lmpath': lmpath, 'tokenizerpath': tk_path}
#     with open(hparams_file) as fin:
#         hparams = load_hyperpyyaml(fin, overrides)
#     asr_model = EncoderDecoderASR.from_hparams(source=model, savedir=model)
#     print("data",data)
#     try:
#         res = asr_model.transcribe_file(path=data)
#     except:
#         print("not done")
#         return  {"text":'result###',"path_audio":"http://10.0.68.100:5019/"+data}
#     del asr_model
#     gc.collect()
#     torch.cuda.empty_cache()
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#     print('result###', res)
#     return {"text":'result###'+res,"path_audio":"http://10.0.68.100:5019/"+data}

# @app.route('/inferfilepath',methods=['POST'])
# def inferfilepath():
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     model = request.form.get('model','dataset/audio')
#     data = request.form.get('data','dataset/excel')
#     if "http://10.0.68.100" in data:
#         data=data
#     data=data.replace("http://10.0.68.100:5019/","")
#     name_res = model
#     asrpath = os.path.join(name_res, 'asr.ckpt')
#     lmpath = os.path.join(name_res, 'lm.ckpt')
#     tk_path = os.path.join(name_res, 'tokenizer.model')
#     hparams_file = 'pretrained/hyperparams.yaml'
#     overrides = {'asrpath': asrpath, 'lmpath': lmpath, 'tokenizerpath': tk_path}
#     with open(hparams_file) as fin:
#         hparams = load_hyperpyyaml(fin, overrides)
#     asr_model = EncoderDecoderASR.from_hparams(source=model, savedir=model)
#     print("data",data)
#     try:
#         res = asr_model.transcribe_file(path=data)
#     except:
#         print("not done")
#         return  {"text":'result###',"path_audio":"http://10.0.68.100:5019/"+data}
#     del asr_model
#     gc.collect()
#     torch.cuda.empty_cache()
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#     print('result###', res)
#     return {"text":'result###'+res,"path_audio":"http://10.0.68.100:5019/"+data}


@app.route('/trainingApi',methods=['POST', 'GET'])
# if methods
def trainingApi():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # audio_dataset = request.form.get('audio_dataset','dataset/audio')
    excel_dataset = request.form.get('excel_dataset',default=r'C:\HungDangT04\Finetune_whisper\dataset\final_script.csv')
    # useFolderinput = request.form.get('useFolderinput','0')
    # print("useFolderinput",useFolderinput)
    # if useFolderinput=="0":
    #     file = request.files['file']
    #     if not file:
    #         return jsonify({'message': 'No file uploaded'}), 400
    #     if file.filename == '':
    #         return jsonify({'message': 'No file selected'}), 400
    #     print("file",file)
    #     if file:
    #         time_dir=time.time()
    #         path_exel=os.path.join(r"C:\HungDangT04\speechbrain_stt_training\excel_data",str(time_dir))
    #         os.makedirs(path_exel)
    #         excel_dataset=os.path.join(path_exel,file.filename)
    #         print("excel_dataset",excel_dataset)
    #         file.save(excel_dataset)
    #         excel_dataset=path_exel
    
    # vocab_size = request.form.get('vocab_size','489')
    # batch_size = request.form.get('batch_size','1')
    # countdata = request.form.get('Countvalue','1')
    per_device_train_batch_size = request.form.get('per_device_train_batch_size','16')
    learning_rate = request.form.get('learning_rate','1e-5')
    warmup_steps = request.form.get('warmup_steps','500')
    max_steps = request.form.get('max_steps','4000')
    per_device_eval_batch_size = request.form.get('per_device_eval_batch_size','8')
    generation_max_length = request.form.get('generation_max_length','225')
    save_steps = request.form.get('save_steps','2000')
    eval_steps = request.form.get('eval_steps','2000')
    logging_steps = request.form.get('eval_steps','25')
    txtfile = request.form.get('txtfile','txtfile.txt')

    lang = request.form.get('lang','vi')
    idx=request.form.get("idx")
    epoch = request.form.get('epoch','1')
    # print("audio_dataset",audio_dataset)
    print("excel_dataset",excel_dataset)
    # os.makedirs(audio_dataset, exist_ok=True)
    txtfile=os.path.join(r"txtfile",str(time.time())+".txt")
    print("txtfile",txtfile)
    datecreate = datetime.datetime.now()
    label_file = os.path.join(excel_dataset, 'final_script.csv')
    # string_run=r"C:\Users\TNADMIN\AppData\Local\Programs\Python\Python37\python.exe C:\HungDangT04\speechbrain_stt_training\training_quang.py"+" --idx "+idx+" --audio_dataset "+audio_dataset+\
    # " --excel_dataset "+excel_dataset+ " --useFolderinput "+useFolderinput+" --vocab_size "+vocab_size+" --batch_size "+batch_size+" --epoch "+epoch+" --txtfile {}".format(txtfile)
    venv_path = os.path.join(os.path.dirname(__file__), r"venv/Scripts/python.exe")
    training_path = os.path.join(os.path.dirname(__file__), r"training_quang.py")
    argument = f"--excel_dataset {excel_dataset} --per_device_train_batch_size {per_device_train_batch_size} --learning_rate {learning_rate} --warmup_steps {warmup_steps} --max_steps {max_steps} --per_device_eval_batch_size {per_device_eval_batch_size} --generation_max_length {generation_max_length} --save_steps {save_steps} --eval_steps {eval_steps} --logging_steps {logging_steps} --txtfile {txtfile}"
    string_run = f"{venv_path} {training_path} {argument}"
    result =os.system(string_run)
    
    return txtfile


    # print("Asdasd")
    # if result == 0:
    #     print("Command executed successfully")
    #     if  os.path.exists(txtfile):
    #         f=open(txtfile,"r")
    #         line=f.read()
    #         Pathfolder=line.replace("model###", "")
    #         path="C:\HungDangT04\speechbrain_stt_training"
    #         Pathfolder=r'pretrained\1694959444.4241817'
    #         pathnew=os.path.join(path,Pathfolder)

    #         Insert_test = "INSERT INTO  " + "Audio_ModelTraing"+ " ( [Pathfolder],[datecreate],[countdata],[typeProcess],[lang],[accurate]) \
    #                                         VALUES ('{}','{}','{}','{}','{}','{}')".format(pathnew, datecreate,countdata,0,lang,0)  

            
    #         print("Insert_test",Insert_test)
    #         cursor.execute(Insert_test)
    #         return line
    # else:
    #     print("Command failed with error code", result)

    # return line

# @app.route('/trainingApiaccent',methods=['POST'])
# def trainingApiaccent():
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     audio_dataset = request.form.get('audio_dataset','dataset/audio')
#     excel_dataset = request.form.get('excel_dataset','dataset/excel')
#     useFolderinput = request.form.get('useFolderinput','0')
#     print("useFolderinput",useFolderinput)
#     if useFolderinput=="0":
#         file = request.files['file']
#         if not file:
#             return jsonify({'message': 'No file uploaded'}), 400
#         if file.filename == '':
#             return jsonify({'message': 'No file selected'}), 400
#         print("file",file)
#         if file:
#             time_dir=time.time()
#             path_exel=os.path.join(r"C:\HungDangT04\speechbrain_stt_training\excel_data",str(time_dir))
#             os.makedirs(path_exel)
#             excel_dataset=os.path.join(path_exel,file.filename)
#             print("excel_dataset",excel_dataset)
#             file.save(excel_dataset)
#             excel_dataset=path_exel
    
 
#     vocab_size = request.form.get('vocab_size','489')
#     batch_size = request.form.get('batch_size','1')
#     n_classes= request.form.get('n_classes','7')
#     epoch = request.form.get('epoch','1')
#     print("audio_dataset",audio_dataset)
#     print("excel_dataset",excel_dataset)
#     os.makedirs(audio_dataset, exist_ok=True)
#     txtfile=os.path.join(r"C:\HungDangT04\speechbrain_stt_training\txtfile",str(time.time())+".txt")
#     print("txtfile",txtfile)
#     label_file = os.path.join(excel_dataset, 'final_script.csv')
#     string_run=r"cd C:\TueT04\accent_id\speechbrain\templates\speaker_id\ & C:\Users\TNADMIN\anaconda3\envs\accent_id\python.exe   C:\TueT04\accent_id\speechbrain\templates\speaker_id\train_v2.py"+" --audio_dataset "+audio_dataset+\
#     " --excel_dataset "+excel_dataset+ " --batch_size "+batch_size+" --epoch "+epoch+" --txtfile {}".format(txtfile)
#     result =os.system(string_run)
#     print("result:", result)
#     if result == 0:
#         print("Command executed successfully")
#         if  os.path.exists(txtfile):
#             f=open(txtfile,"r")
#             line=f.read()
            
#             return line
#     else:
#         print("Command failed with error code", result)
       
    
#     return line


# @app.route('/killid', methods=['POST'])
# def killid():

    
#     idx=request.form.get("idx")
#     f = open(idx, "r")
#     while True:
#         value=f.readline()
#         try:
#             os.kill(int(value), signal.SIGTERM)
#         except:
#             print("")
#         if value=="":
#             break

  
    

#     return f"kill"

# @app.route('/InferApiaccent',methods=['POST'])
# def InferApiaccent():
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     model = request.form.get('model','dataset/audio')
#     data = request.form.get('audio_file','dataset/excel')
#     txtfile = request.form.get('txtfile','txtfile.txt')
    
#     if "http://10.0.68.100" in data:
#         data=data
#     else:
#         test_file = request.files['file']
#         filename = test_file.filename
#         if filename.endswith('.wav'):
#             time_now = dt.datetime.now()
#             # time_now=dt.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
#             time_now=dt.datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-4]
#             new_file_name = str(time_now)+'_'+secure_filename(test_file.filename)
#             file_path = os.path.join('predict', new_file_name)
#             test_file.save(file_path)
            
#     file_path1=os.path.join("C:\HungDangT04\speechbrain_stt_training",file_path)
 
    
#     txtfile=os.path.join(r"C:\HungDangT04\speechbrain_stt_training\txtfile",str(time.time())+".txt")
#     print("txtfile",txtfile)
#     print("file_path",file_path)
#     print("model ",model)
#     string_run=r"cd C:\TueT04\accent_id\speechbrain\templates\speaker_id & C:\Users\TNADMIN\anaconda3\envs\accent_id\python.exe   inference_v3.py"+" --audio_file "+file_path1+\
#     " --model_path "+model+ " --txtfile {}".format(txtfile)

#     result =os.system(string_run)
#     if result == 0:
#         print("Command executed successfully")
#         if  os.path.exists(txtfile):
#             f=open(txtfile,"r")
#             line=f.read()
#             print("line",line)
#             return {"text":line,"path_audio":"http://10.0.68.100:5019/"+file_path}
#     else:
#         print("Command failed with error code", result)


# @app.teardown_appcontext
# def cleanup(resp_or_exc):
#     # Giải phóng bộ nhớ của mô hình
#     # Giải phóng bộ nhớ của GPU
#     torch.cuda.empty_cache()
#     # Giải phóng bộ nhớ khác được sử dụng trong ứng dụng
#     gc.collect()
    # return resp_or_exc
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5019)


  


