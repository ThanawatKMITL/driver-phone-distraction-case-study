import os
import subprocess

custom_train_dir = r"D:\Documents\University\Adv Proj\Train Model\SSD\V1"
pipeline_config = r"D:\Documents\University\Adv Proj\Model\tf_workspace\my_ssd_mobnet\pipeline.config"
train_script = r"D:\Documents\University\Adv Proj\Model\tf_workspace\models\research\object_detection\model_main_tf2.py"

os.environ['XLA_FLAGS'] = r"--xla_gpu_cuda_data_dir=D:\Terminal\ancondaEnviroment\driver_ai\Library"
os.makedirs(custom_train_dir, exist_ok=True)

print("Starting training sequence")

command = [
"python",
train_script,
"--model_dir=" + custom_train_dir,
"--pipeline_config_path=" + pipeline_config
]

subprocess.run(command)