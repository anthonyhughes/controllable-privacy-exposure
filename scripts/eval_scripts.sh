# Claude Pipelines
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/ollama_pipeline.py -t brief_hospital_course &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/ollama_pipeline.py -t discharge_instructions &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/claude_pipeline.py -t brief_hospital_course -m claude-3-5-sonnet-20240620 &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/claude_pipeline.py -t cnn -m claude-3-5-sonnet-20240620 &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/claude_pipeline.py -t cnn -m claude-3-5-sonnet-20240620 > cnn-sonnet.log 2>&1&

# Eval Pielines CNN
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m gpt-4o-mini &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m llama-3-8b-Instruct-bnb-4bit &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m mistral-7b-instruct-v0.3-bnb-4bit &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m claude-3-5-sonnet-20240620 &

# Eval Pielines BHC
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t brief_hospital_course -m gpt-4o-mini &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t brief_hospital_course -m llama-3-8b-Instruct-bnb-4bit &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t brief_hospital_course -m mistral-7b-instruct-v0.3-bnb-4bit &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t brief_hospital_course -m claude-3-5-sonnet-20240620 &

# Eval Pielines DI
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t discharge_instructions -m gpt-4o-mini &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t discharge_instructions -m llama-3-8b-Instruct-bnb-4bit &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t discharge_instructions -m mistral-7b-instruct-v0.3-bnb-4bit &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t discharge_instructions -m claude-3-5-sonnet-20240620 -e utility &

# Eval Pipelines legal contracts
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t legal_court -m gpt-4o-mini -e utility &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t legal_court -m llama-3-8b-Instruct-bnb-4bit -e utility &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t legal_court -m mistral-7b-instruct-v0.3-bnb-4bit -e utility &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t legal_court -m claude-3-5-sonnet-20240620 -e utility &

# Eval Pipelines cnn
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m gpt-4o-mini -e privacy &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m llama-3-8b-Instruct-bnb-4bit -e privacy &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m mistral-7b-instruct-v0.3-bnb-4bit -e privacy &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m claude-3-5-sonnet-20240620 -e privacy &