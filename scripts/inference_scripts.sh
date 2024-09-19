# Inference Pipelines for Llama 70b
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/unsloth_pipeline.py -t cnn -m Meta-Llama-3.1-70B-Instruct-bnb-4bit > llama70b-cnn.log 2>&1&
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t legal_court -m Meta-Llama-3.1-70B-Instruct-bnb-4bit &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t brief_hospital_course -m Meta-Llama-3.1-70B-Instruct-bnb-4bit &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t discharge_instructions -m Meta-Llama-3.1-70B-Instruct-bnb-4bit &