#!/bin/bash

#SBATCH -o log/qqp/prompt_slurm_%j_%a.log
#SBATCH --array=0-9
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2
#SBATCH --qos=high

set -e

domain=$1
# domain='qqp'

train_size='12'

eval_mode=$2
# eval_mode='base'

ft_mode=$3
# ft_mode='st'

mix_rate='0.9'

exp_id=$4
# exp_id=$SLURM_ARRAY_TASK_ID

RED='tput setaf 1'
NC='tput sgr0'

tput setaf 1; printf "\n------------------- Eval Pretrain -------------------\n"
tput sgr0
python eval_prompt.py $domain mt $eval_mode 2 50 pretrain $exp_id

for i in {1..3}
do
    tput setaf 1; printf "\n------------------- Round $i -------------------\n"
    tput sgr0
    python proc_data.py $domain train $train_size $exp_id

    tput setaf 1; printf "\n--------- Prompt_1 ---------\n"
    tput sgr0
    python prompt_cst.py $domain prompt_1 $train_size $ft_mode $exp_id
    # python mix_model.py $domain $mix_rate
    python eval_prompt.py $domain mt $eval_mode 2 50 self_train $exp_id
    
    if [[ "$ft_mode" == "ft" ]]; then
        continue
    fi

    tput setaf 1; printf "\n--------- Prompt_2 ---------\n"
    tput sgr0
    python prompt_cst.py $domain prompt_2 $train_size $ft_mode $exp_id
    # python mix_model.py $domain $mix_rate
    python eval_prompt.py $domain mt $eval_mode 2 50 self_train $exp_id

    tput setaf 1; printf "\n--------- Prompt_joint ---------\n"
    tput sgr0
    python prompt_cst.py $domain prompt_joint $train_size $ft_mode $exp_id
    # python mix_model.py $domain $mix_rate
    python eval_prompt.py $domain mt $eval_mode 2 50 self_train $exp_id
done

# tput setaf 1; echo '--------- Adv Evaluation ---------'
# tput sgr0
# python eval_adv_offline.py $domain large train

# tput setaf 1; echo '--------- Base Evaluation ---------'
# tput sgr0
# python eval_base_offline.py $domain large train