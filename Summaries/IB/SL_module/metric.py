import re
import os
import csv
folder_path="/home/jay/MR.HiSum/Summaries/IB/SL_module/multi/cib/"
def compute_average(tmp_list):
    return sum(tmp_list)/len(tmp_list)
beta_list=[1e-6]
for beta in beta_list:
    total_data=[]
    for i in sorted(os.listdir(folder_path)):
        #print(i)
        sub_path=os.path.join(folder_path,i,str(beta))
        try:
            file_path=os.path.join(sub_path,os.listdir(sub_path)[0],"results.txt")
            with open(file_path, 'r') as file:
                print(f"open the file: {file_path}")
                content = file.readlines()
            
            # 初始化變數
            best_f1score_model= [] 
            best_mAP50_model= []
            best_mAP15_model= []
            current_model = None
            
            # 正則表達式匹配模型名稱與數據
            model_pattern = re.compile(r'Testing on .*?(best_f1score_model|best_mAP50_model|best_mAP15_model)')
            fscore_pattern = re.compile(r'Test F-score\s+([\d\.]+)')
            map50_pattern = re.compile(r'Test MAP50\s+([\d\.]+)')
            map15_pattern = re.compile(r'Test MAP15\s+([\d\.]+)')
            
            # 解析文件內容
            for i in range(len(content)):
                line=content[i]
                if model_pattern.search(line):
                    #print(line)
                    #print(content[i+1])
                    F1=content[i+1].split(' ')[-1]
                    best_f1score_model.append(float(F1))
                    #print(content[i+2])
                    map50=content[i+2].split(' ')[-1]
                    best_mAP50_model.append(float(map50))
                    #print(content[i+3])
                    map15=content[i+3].split(' ')[-1]
                    best_mAP15_model.append(float(map15))
            #print(best_f1score_model)
            #print(best_mAP50_model)
            #print(best_mAP15_model)
            # 計算平均值
            
            average_list = [compute_average(best_f1score_model),compute_average(best_mAP50_model)*100, compute_average(best_mAP15_model)*100]
            print(average_list)
            total_data.append(average_list)
        except:
            print(f"{sub_path} has no result.txt")
            continue
    output_path=os.path.join(folder_path,f"result_beta_{beta}.csv")
    for i in total_data:
        with open(output_path, "a", newline="") as file:
            writer = csv.writer(file)
            for item in i:
                writer.writerow([item])# 這樣寫入單行