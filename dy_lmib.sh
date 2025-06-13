#!/bin/bash
#python main_dy_lmib.py --train True --model SL_module --batch_size 60 --modal multi --device 1 --lr 0.05 --type dylib --vbeta 1e-01 --abeta 1e-01 --epochs 1 --tag mr_test_dylmib --path dataset/mr_hisum_split.json
time_tag="05192144"
type="dylib_v0"
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
  "1e-01"
  "1e-02"
  "1e-03"
  "1e-04"
 "1e-05"
 "1e-06"

)
# 第一個指令
i=150
echo "Running for ${type}, epochs: ${i}"

for value in "${beta_list[@]}"
do
    for path in "${datasets[@]}"
    do
        # 從 JSON 文件名提取 cate 類別
        cate=$(basename "${path}" | sed -E 's/_split\.json$//;s/[ &()]/_/g')
        
        echo "Processing dataset: ${path} (Category: ${cate})"
        python main_dy_lmib.py --train True --model SL_module --batch_size 40 --modal ${modal} --device 1 --lr 0.05 --type ${type} --abeta ${value} --vbeta 0 --epochs ${i} --tag ${cate}_${type}_ep${i}_${time_tag}_abeta${value}_vbeta0 --path ${path}
    done
done


echo "所有指令執行完畢。"
