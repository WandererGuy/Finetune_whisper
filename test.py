import os

excel_dataset = 'dataset/final_script.csv'
vocab_size = '489'
batch_size = 1
countdata = '1'
per_device_train_batch_size = '16'
learning_rate = '1e-5'
warmup_steps = '5'
max_steps = '40'
per_device_eval_batch_size = '8'
generation_max_length = '225'
save_steps = '20'
eval_steps = '20'
logging_steps = '2'
txtfile = 'txtfile.txt'

venv_path = os.path.join(os.path.dirname(__file__), r"venv/Scripts/python.exe")
training_path = os.path.join(os.path.dirname(__file__), r"training_quang.py")
argument = f"--excel_dataset {excel_dataset} --per_device_train_batch_size {per_device_train_batch_size} --learning_rate {learning_rate} --warmup_steps {warmup_steps} --max_steps {max_steps} --per_device_eval_batch_size {per_device_eval_batch_size} --generation_max_length {generation_max_length} --save_steps {save_steps} --eval_steps {eval_steps} --logging_steps {logging_steps} --txtfile {txtfile}"
string_run = f"{venv_path} {training_path} {argument}"

print(string_run)

