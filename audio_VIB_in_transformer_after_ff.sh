#!/bin/bash
# parser.add_argument('--model', type = str, default = 'MLP', help = 'the name of the model')
# parser.add_argument('--epochs', type = int, default = 200, help = 'the number of training epochs')
# parser.add_argument('--lr', type = float, default = 5e-5, help = 'the learning rate')
# parser.add_argument('--l2_reg', type = float, default = 1e-4, help = 'l2 regularizer')
# parser.add_argument('--dropout_ratio', type = float, default = 0.5, help = 'the dropout ratio')
# parser.add_argument('--batch_size', type = int, default = 256, help = 'the batch size')
# parser.add_argument('--tag', type = str, default = 'dev', help = 'A tag for experiments')
# parser.add_argument('--ckpt_path', type = str, default = None, help = 'checkpoint path for inference or weight initialization')
# parser.add_argument('--train', type=str2bool, default='true', help='when use Train')
# parser.add_argument('--path', type=str, default='dataset/metadata_split.json', help='path')
# parser.add_argument('--device', type=str, default='0', help='gpu')
# parser.add_argument('--modal', type=str, default='visual', help='visual,audio,multi')
# parser.add_argument('--beta', type=float, default=0, help='beta')
# parser.add_argument('--type',type = str, default='base', help='base,ib,cib,eib,lib')#cib,eib,lib
time_tag="07091636" #date
type="ib" #'base,ib,cib,eib,lib'
#1e-06
# 定義所有 dataset 路徑
datasets=(
#   "dataset/Hobbies_Leisure_split.json"
#   "dataset/(Unknown)_split.json"
#   "dataset/Business_Industrial_split.json"
#   "dataset/Pets_Animals_split.json"
#   "dataset/Real_Estate_split.json"
#   "dataset/Shopping_split.json"
#   "dataset/News_split.json"
#   "dataset/Food_Drink_split.json"
#   "dataset/Home_Garden_split.json"
#   "dataset/Books_Literature_split.json"
#   "dataset/Beauty_Fitness_split.json"
#   "dataset/Travel_split.json"
#   "dataset/Health_split.json"
#   "dataset/Law_Government_split.json"
#   "dataset/Games_split.json"
#   "dataset/Computers_Electronics_split.json"
#   "dataset/Arts_Entertainment_split.json"
#   "dataset/Science_split.json"
#   "dataset/People_Society_split.json"
#  "dataset/Reference_split.json"
#   "dataset/Internet_Telecom_split.json"
#   "dataset/Autos_Vehicles_split.json"
#   "dataset/Sports_split.json"
#   "dataset/Jobs_Education_split.json"
#   "dataset/Finance_split.json"
  "dataset/mr_hisum_split.json"
)
test_datasets=(
#"dataset/Finance_split.json"
#"dataset/mr_hisum_split.json"
#"dataset/mr_hisum_split.json"
#dataset/Home_Garden_split.json
#"dataset/tvsum/tvsum_split.json"
#"dataset/Games_split.json"
)
beta_list=(
# "0"
"1e-10"
"1e-09"
"1e-08"
"1e-07"
"1e-06"
"1e-05"
"1e-04"
"1e-03"
"1e-02"
"1e-01"
"1"
"10"
# "0"
# "1e-01"
)
# 第一個指令
i=100
path="dataset/mr_hisum_split.json"
echo "Running for ${type}, epochs: ${i}"
for value in "${beta_list[@]}"
do
    # 從 JSON 文件名提取 cate 類別
    cate=$(basename "${path}" | sed -E 's/_split\.json$//;s/[ &()]/_/g')
    
    echo "Processing dataset: ${path} (Category: ${cate})"
    python main_audio_VIB_in_tran_after_ff.py --train True --model SL_module --batch_size 100 --modal audio --device 2 --type ${type} --beta ${value} --epochs ${i} --tag VIBintranAfterFF_in_transformer_epoch${i}_${time_tag} --path ${path}
done


echo "所有指令執行完畢。"
#python main_visual_VIB.py --train True --model SL_module --batch_size 128 --modal visual --device 0 --type ib --beta 0 --epochs 150 --tag visual_VIB_mr_ib_ep_150_03120915 --path "dataset/mr_hisum_split.json"
#fuser -v /dev/nvidia* | awk '{for(i=1; i<=NF; i++) if($i ~ /^[0-9]+$/) print $i}' | xargs -I{} kill -9 {} 
