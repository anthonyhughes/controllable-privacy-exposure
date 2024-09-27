# Inference Pipelines for Llama 70b
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/unsloth_pipeline.py -t cnn -m Meta-Llama-3.1-70B-Instruct-bnb-4bit > llama70b-cnn.log 2>&1&
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/unsloth_pipeline.py -t discharge_instructions -m Meta-Llama-3.1-70B-Instruct-bnb-4bit > llama70b-di.log 2>&1&
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/unsloth_pipeline.py -t brief_hospital_course -m Meta-Llama-3.1-70B-Instruct-bnb-4bit > llama70b-bhi.log 2>&1&
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/unsloth_pipeline.py -t legal_court -m Meta-Llama-3.1-70B-Instruct-bnb-4bit > llama70b-lc-sani-summ.log 2>&1&

# Inference pipelines for mistral 8b
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/unsloth_pipeline.py -t discharge_instructions -m mistral-7b-instruct-v0.3-bnb-4bit > mistral8b-di.log 2>&1&
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/unsloth_pipeline.py -t cnn -m mistral-7b-instruct-v0.3-bnb-4bit > mistral8b-cnn.log 2>&1&
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/unsloth_pipeline.py -t legal_court -m mistral-7b-instruct-v0.3-bnb-4bit > mistral8b-lc-sani-summ.log 2>&1&
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/unsloth_pipeline.py -t brief_hospital_course -m mistral-7b-instruct-v0.3-bnb-4bit > mistral8b-bhi-sani-summ.log 2>&1&

# Inference pipelines for llama 3.1 8b
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/unsloth_pipeline.py -t discharge_instructions -m llama-3-8b-Instruct-bnb-4bit > llama8b-di.log 2>&1&
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/unsloth_pipeline.py -t cnn -m llama-3-8b-Instruct-bnb-4bit > llama8b-cnn.log 2>&1&
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/unsloth_pipeline.py -t legal_court -m llama-3-8b-Instruct-bnb-4bit > llama8b-lc-sani-summ.log 2>&1&
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/unsloth_pipeline.py -t brief_hospital_course -m llama-3-8b-Instruct-bnb-4bit > llama8b-bhi-sani-summ.log 2>&1&
