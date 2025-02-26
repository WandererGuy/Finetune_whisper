from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import os
import whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_path = ''

processor = WhisperProcessor.from_pretrained("processor-pretrained")
model = WhisperForConditionalGeneration.from_pretrained(model_path)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="vietnamese", task="transcribe")

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './data_folder'
@app.route('/speech_to_text_medium', methods=['GET', 'POST'])
def speech_to_text_medium():
    upload_file = request.files['file']
    language = request.form['language']
    upload_file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], upload_file.filename))
    app.config['WAV'] = upload_file.filename

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], app.config['WAV'])
    
    options = {
    "language": language, # input language, if omitted is auto detected
    "task": "transcribe" # or "transcribe" if you just want transcription
}
    res = model_medium.transcribe(file_path, **options)

    return {'result': res["text"]}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='1516', debug=False)
