#!/bin/bash

time_tag="06220248"
type="dycmib"
modal="multi"
#1e-06
# 定義所有 dataset 路徑
datasets=(
 #  "dataset/Hobbies_Leisure_split.json"
 #  "dataset/(Unknown)_split.json"
 #  "dataset/Business_Industrial_split.json"
 #  "dataset/Pets_Animals_split.json"
 #  "dataset/Real_Estate_split.json"
 #  "dataset/Shopping_split.json"
 #  "dataset/News_split.json"
 #  "dataset/Food_Drink_split.json"
 #  "dataset/Home_Garden_split.json"
 #  "dataset/Books_Literature_split.json"
 #  "dataset/Beauty_Fitness_split.json"
 #  "dataset/Travel_split.json"
 #  "dataset/Health_split.json"
 #  "dataset/Law_Government_split.json"
 #  "dataset/Games_split.json"
 #  "dataset/Computers_Electronics_split.json"
 #  "dataset/Arts_Entertainment_split.json"
 #  "dataset/Science_split.json"
 #  "dataset/People_Society_split.json"
 # "dataset/Reference_split.json"
 #  "dataset/Internet_Telecom_split.json"
 #  "dataset/Autos_Vehicles_split.json"
 #  "dataset/Sports_split.json"
 #  "dataset/Jobs_Education_split.json"
 #  "dataset/Finance_split.json"
"dataset/mr_hisum_split.json"
)
test_datasets=(
"dataset/Finance_split.json"
#"dataset/mr_hisum_split.json"
#"dataset/mr_hisum_split.json"
#dataset/Home_Garden_split.json
#"dataset/tvsum/tvsum_split.json"
#"dataset/Games_split.json"
)
beta_list=(
# "10"
# "1"
# "0"
#"1e-01"
#"1e-02"
"1e-03"
"0.0009"
"0.0008"
"0.0007"
"0.0006"
"0.0005"
"0.0004"
"0.0003"
"0.0002"
"1e-04"
"0.00009"
"0.00008"
"0.00007"
"0.00006"
"0.00005"
"0.00004"
"0.00003"
"0.00002"
"1e-05"
#"1e-06"

)
# 设置环境变量，强制只使用GPU 0
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#  2  Tesla V100-SXM2-16GB
# 第一個指令
i=150
path="dataset/mr_hisum_split.json"
batch_size=64
echo "Running for ${type}, epochs: ${i} on GPU 0 only"

for value in "${beta_list[@]}"
do
    # 從 JSON 文件名提取 cate 類別
    cate=$(basename "${path}" | sed -E 's/_split\.json$//;s/[ &()]/_/g')
    
    echo "Processing dataset: ${path} (Category: ${cate})"
    python main_dy_cmib_backup.py --train True  --batch_size ${batch_size} --device 1 --modal ${modal} --lr 0.05 --type ${type} --vbeta ${value} --abeta ${value} --mbeta ${value} --epochs ${i} --tag ${cate}_${type}_ep${i}_${time_tag}_samebeta_${value} --path ${path}

done
#python main_dy_cmib_backup.py --train True  --batch_size ${batch_size} --device 1 --modal ${modal} --lr 0.05 --type ${type} --vbeta 0.0008 --abeta 0.0008 --mbeta 0.0008 --epochs ${i} --tag ${cate}_${type}_ep${i}_${time_tag}_samebeta_0.0008 --path ${path} 
#python main_dy_cmib_backup.py --train True  --batch_size ${batch_size} --device 1 --modal ${modal} --lr 0.05 --type ${type} --vbeta 0.0004 --abeta 0.0004 --mbeta 0.0004 --epochs ${i} --tag ${cate}_${type}_ep${i}_${time_tag}_samebeta_0.0004 --path ${path}
#python main_dy_cmib_backup.py --train True  --batch_size ${batch_size} --device 1 --modal ${modal} --lr 0.05 --type ${type} --vbeta 0.00009 --abeta 0.00009 --mbeta 0.00009 --epochs ${i} --tag ${cate}_${type}_ep${i}_${time_tag}_samebeta_0.00009 --path ${path}
#python main_dy_cmib_backup.py --train True  --batch_size ${batch_size} --device 0 --modal ${modal} --lr 0.05 --type ${type} --vbeta 0.00005 --abeta 0.00005 --mbeta 0.00005 --epochs ${i} --tag ${cate}_${type}_ep${i}_${time_tag}_samebeta_0.00005 --path ${path}
#python main_dy_cmib_backup.py --train True  --batch_size ${batch_size} --device 0 --modal ${modal} --lr 0.05 --type ${type} --vbeta 0.00001 --abeta 0.00001 --mbeta 0.00001 --epochs ${i} --tag ${cate}_${type}_ep${i}_${time_tag}_samebeta_0.00001 --path ${path}
# python main_dy_cmib_backup.py --train True  --batch_size ${batch_size} --device 0 --modal ${modal} --lr 0.05 --type ${type} --vbeta 0.0003 --abeta 0.0003 --mbeta 0.0003 --epochs ${i} --tag ${cate}_${type}_ep${i}_${time_tag}_samebeta_0.0003 --path ${path}
# python main_dy_cmib_backup.py --train True  --batch_size ${batch_size} --device 0 --modal ${modal} --lr 0.05 --type ${type} --vbeta 0.00008 --abeta 0.00008 --mbeta 0.00008 --epochs ${i} --tag ${cate}_${type}_ep${i}_${time_tag}_samebeta_0.00008 --path ${path}
# python main_dy_cmib_backup.py --train True  --batch_size ${batch_size} --device 0 --modal ${modal} --lr 0.05 --type ${type} --vbeta 0.00004 --abeta 0.00004 --mbeta 0.00004 --epochs ${i} --tag ${cate}_${type}_ep${i}_${time_tag}_samebeta_0.00004 --path ${path}
# python main_dy_cmib_backup.py --train True  --batch_size ${batch_size} --device 0 --modal ${modal} --lr 0.05 --type ${type} --vbeta 0.0002 --abeta 0.0002 --mbeta 0.0002 --epochs ${i} --tag ${cate}_${type}_ep${i}_${time_tag}_samebeta_0.0002 --path ${path}
# python main_dy_cmib_backup.py --train True  --batch_size ${batch_size} --device 0 --modal ${modal} --lr 0.05 --type ${type} --vbeta 0.00007 --abeta 0.00007 --mbeta 0.00007 --epochs ${i} --tag ${cate}_${type}_ep${i}_${time_tag}_samebeta_0.00007 --path ${path}
# python main_dy_cmib_backup.py --train True  --batch_size ${batch_size} --device 0 --modal ${modal} --lr 0.05 --type ${type} --vbeta 0.00003 --abeta 0.00003 --mbeta 0.00003 --epochs ${i} --tag ${cate}_${type}_ep${i}_${time_tag}_samebeta_0.00003 --path ${path}
echo "所有指令執行完畢。"
# python main_dy_cmib_backup.py --train True  --batch_size 64 --device 1 --modal multi --lr 0.005 --type dycmib --vbeta 0.001 --abeta 0.001 --mbeta 0.001 --epochs 150 --tag test_samebeta_0.001 --path "dataset/mr_hisum_split.json"