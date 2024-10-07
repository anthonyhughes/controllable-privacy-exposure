# Eval Pielines CNN
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m gpt-4o-mini &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m llama-3-8b-Instruct-bnb-4bit &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m mistral-7b-instruct-v0.3-bnb-4bit &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m claude-3-5-sonnet-20240620 &

# Eval Pielines BHC - Utility
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t brief_hospital_course -m gpt-4o-mini -e utility &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t brief_hospital_course -m llama-3-8b-Instruct-bnb-4bit -e utility &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t brief_hospital_course -m mistral-7b-instruct-v0.3-bnb-4bit -e utility &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t brief_hospital_course -m claude-3-5-sonnet-20240620 -e utility &

# Eval Pipelines DI - Utility
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t discharge_instructions -m gpt-4o-mini &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t discharge_instructions -m llama-3-8b-Instruct-bnb-4bit &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t discharge_instructions -m mistral-7b-instruct-v0.3-bnb-4bit &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t discharge_instructions -m claude-3-5-sonnet-20240620 -e utility &

# Eval Pipelines legal contracts
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t legal_court -m gpt-4o-mini -e utility &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t legal_court -m llama-3-8b-Instruct-bnb-4bit -e utility &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t legal_court -m mistral-7b-instruct-v0.3-bnb-4bit -e utility &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t legal_court -m claude-3-5-sonnet-20240620 -e utility &

# Eval Privacy Pipelines - CNN
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m gpt-4o-mini -e privacy &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m llama-3-8b-Instruct-bnb-4bit -e privacy &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m mistral-7b-instruct-v0.3-bnb-4bit -e privacy &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m claude-3-5-sonnet-20240620 -e privacy &

# Eval Privacy Pipelines - Discharge Summaries
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t discharge_instructions -m gpt-4o-mini -e privacy &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t discharge_instructions -m llama-3-8b-Instruct-bnb-4bit -e privacy &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t discharge_instructions -m mistral-7b-instruct-v0.3-bnb-4bit -e privacy &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t discharge_instructions -m claude-3-5-sonnet-20240620 -e privacy &

# All tasks utility
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t all -m gpt-4o-mini -e utility &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t all -m llama-3-8b-Instruct-bnb-4bit -e utility &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t all -m mistral-7b-instruct-v0.3-bnb-4bit -e utility &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t all -m claude-3-5-sonnet-20240620 -e utility &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t all -m Meta-Llama-3.1-70B-Instruct-bnb-4bit -e utility &

# All tasks utility
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t all -m gpt-4o-mini -e privacy &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t all -m llama-3-8b-Instruct-bnb-4bit -e privacy &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t all -m mistral-7b-instruct-v0.3-bnb-4bit -e privacy &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t all -m claude-3-5-sonnet-20240620 -e privacy &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t all -m Meta-Llama-3.1-70B-Instruct-bnb-4bit -e privacy &


PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t all -m mistral-7b-instruct-v0.3-bnb-4bit -e privacy -st all &
PYTHONPATH=/home/acp23ajh/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t all -m gpt-4o-mini -e privacy -st all &

PYTHONPATH=/Users/anthonyhughes/PhD/controllable-privacy-exposure nohup python pipelines/eval_pipeline.py -t all -m llama-3-8b-Instruct-bnb-4bit -e privacy -st all > llama8b-pii.log 2>&1&
PYTHONPATH=/Users/anthonyhughes/PhD/controllable-privacy-exposure nohup python pipelines/eval_pipeline.py -t all -m claude-3-5-sonnet-20240620 -e privacy -st all > claude-pii.log 2>&1&
PYTHONPATH=/Users/anthonyhughes/PhD/controllable-privacy-exposure nohup python pipelines/eval_pipeline.py -t all -m Meta-Llama-3.1-70B-Instruct-bnb-4bit -e privacy -st all > llama70b-pii.log 2>&1&


PYTHONPATH=/Users/anthonyhughes/PhD/controllable-privacy-exposure nohup python pipelines/eval_pipeline.py -t all -m llama-3-8b-Instruct-bnb-4bit -e privacy -st _sani_summ &
PYTHONPATH=/Users/anthonyhughes/PhD/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t discharge_instructions -m Meta-Llama-3.1-70B-Instruct-bnb-4bit -e utility -st _sani_summ &

PYTHONPATH=/Users/anthonyhughes/PhD/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m llama-3-8b-Instruct-bnb-4bit -e utility -st all &
PYTHONPATH=/Users/anthonyhughes/PhD/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m Meta-Llama-3.1-70B-Instruct-bnb-4bit -e utility -st all &
PYTHONPATH=/Users/anthonyhughes/PhD/controllable-privacy-exposure/ nohup python pipelines/eval_pipeline.py -t cnn -m claude-3-5-sonnet-20240620 -e utility -st all &
