virtualenv venv --python=python3.9.9
pip install -r .\requirements.txt
pip install accelerate -U
pip install jiwer
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade accelerate
pip show transformers accelerate
pip install --upgrade --quiet datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio

& C:/Users/TNADMIN/AppData/Local/Programs/Python/Python38/python.exe c:\Users\TNADMIN\.vscode\extensions\ms-python.python-2025.0.0-win32-x64\python_files\printEnvVariablesToFile.py c:\Users\TNADMIN\.vscode\extensions\ms-python.python-2025.0.0-win32-x64\python_files\deactivate\powershell\envVars.txt
.\venv\Scripts\accelerate.exe