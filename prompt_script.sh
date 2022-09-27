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
train_mode='plabel' # plabel, proto
eval_model='sc' # sc, proto

exp_id=$4
base_exp_log_name='base'
exp_log_name='gnc-gm'
# exp_id=$SLURM_ARRAY_TASK_ID

RED='tput setaf 1'
NC='tput sgr0'

tput setaf 1; printf "\n------------------- Eval Pretrain -------------------\n"
tput sgr0

python eval_prompt.py $domain mt $eval_mode 2 50 pretrain $exp_id $eval_model pretrain

touch log/$domain/exp_$exp_id\_$base_exp_log_name.json
touch log/$domain/exp_$exp_id\_$exp_log_name.json

rm log/$domain/exp_$exp_id\_$base_exp_log_name.json
rm log/$domain/exp_$exp_id\_$exp_log_name.json

touch log/$domain/cal_w_[$exp_id].json
rm log/$domain/cal_w_[$exp_id].json

cp log/qnli/prob_template.json log/$domain/prob_stat_ft.json
cp log/qnli/prob_template.json log/$domain/prob_stat_st.json
cp log/qnli/w_template.json log/$domain/w_stat.json

for i in {1..20}
do
    tput setaf 1; printf "\n------------------- Round $i -------------------\n"
    tput sgr0
    # python proc_data.py $domain train $train_size 0 train
    python proc_data.py $domain train $train_size $exp_id train
    # python proc_data.py $domain train $train_size $exp_id dev

    tput setaf 1; printf "\n--------- Prompt_1 ---------\n"
    tput sgr0
    python prompt_cst.py $domain prompt_1 $train_size ft $exp_id $eval_mode $train_mode
    python eval_prompt.py $domain mt $eval_mode 2 50 self_train $exp_id $eval_model base

    python prompt_cst.py $domain prompt_1 $train_size $ft_mode $exp_id $eval_mode $train_mode
    # python mix_model.py $domain $mix_rate
    python eval_prompt.py $domain mt $eval_mode 2 50 self_train $exp_id $eval_model $exp_log_name
done