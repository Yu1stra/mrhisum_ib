import os
import json
from collections import defaultdict

def create_audio_visual_split():
    # 包含 JSON 文件的目錄
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    
    # 要檢查的分類文件列表（按用戶請求提供）
    categories = [
        "Hobbies_Leisure_split.json",
        "(Unknown)_split.json",
        "Business_Industrial_split.json",
        "Pets_Animals_split.json",
        "Real_Estate_split.json",
        "Shopping_split.json",
        "News_split.json",
        "Food_Drink_split.json",
        "Home_Garden_split.json",
        "Books_Literature_split.json",
        "Beauty_Fitness_split.json",
        "Travel_split.json",
        "Health_split.json",
        "Law_Government_split.json",
        "Games_split.json",
        "Computers_Electronics_split.json",
        "Arts_Entertainment_split.json",
        "Science_split.json",
        "People_Society_split.json",
        "Reference_split.json",
        "Internet_Telecom_split.json",
        "Autos_Vehicles_split.json",
        "Sports_split.json",
        "Jobs_Education_split.json",
        "Finance_split.json",
    ]
    
    # 音頻類別（按用戶要求）
    audio_categories = ["Arts_Entertainment", "Games", "Pets_Animals"]
    
    # 存儲每個視頻 ID 首次出現的分類
    video_first_category = {}
    
    # 存儲原始數據的訓練、驗證和測試集分割
    id_to_split = {}  # 記錄每個視頻ID屬於哪個數據集分割(train/val/test)
    
    # 處理每個分類文件
    for category_file in categories:
        file_path = os.path.join(dataset_dir, category_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 從 train、val 和 test keys 中提取所有視頻 ID
            for key_type in ['train_keys', 'val_keys', 'test_keys']:
                if key_type in data:
                    for video_id in data[key_type]:
                        # 如果這個ID第一次出現，記錄它的分類和數據集分割(train/val/test)
                        if video_id not in video_first_category:
                            category_name = category_file.replace('_split.json', '')
                            video_first_category[video_id] = category_name
                            split_type = key_type  # 例如 'train_keys', 'val_keys', 'test_keys'
                            id_to_split[video_id] = split_type
                    
            print(f"處理 {category_file}: 完成")
                
        except FileNotFoundError:
            print(f"警告: 找不到文件 - {file_path}")
            continue
        except json.JSONDecodeError:
            print(f"警告: JSON 解碼錯誤 - {file_path}")
            continue
    
    # 將視頻ID分為音頻組和視覺組
    audio_ids = {
        'train_keys': [],
        'val_keys': [],
        'test_keys': []
    }
    
    visual_ids = {
        'train_keys': [],
        'val_keys': [],
        'test_keys': []
    }
    
    # 按照數據集分割將視頻分配到音頻或視覺組
    for video_id, category in video_first_category.items():
        split_type = id_to_split[video_id]  # 獲取數據集分割類型（例如 'train_keys'）
        
        if category in audio_categories:
            audio_ids[split_type].append(video_id)
        else:
            visual_ids[split_type].append(video_id)
    
    # 統計音頻和視覺組的數量
    total_audio = sum(len(ids) for ids in audio_ids.values())
    total_visual = sum(len(ids) for ids in visual_ids.values())
    
    # 將結果保存為新的JSON文件
    with open(os.path.join(dataset_dir, 'audio_split.json'), 'w', encoding='utf-8') as f:
        json.dump(audio_ids, f, indent=4)
        
    with open(os.path.join(dataset_dir, 'visual_split.json'), 'w', encoding='utf-8') as f:
        json.dump(visual_ids, f, indent=4)
    
    # 打印結果
    print("\n=== 結果 ===")
    print(f"\n音頻類別 (Arts_Entertainment, Games, Pets_Animals): {total_audio} 個視頻")
    print(f"視覺類別 (其他所有類別): {total_visual} 個視頻")
    print(f"總計: {total_audio + total_visual} 個視頻")
    
    # 打印分割情況
    print("\n音頻類別分割:")
    for split, ids in audio_ids.items():
        print(f"  {split.replace('_keys', '')}: {len(ids)}")
        
    print("\n視覺類別分割:")
    for split, ids in visual_ids.items():
        print(f"  {split.replace('_keys', '')}: {len(ids)}")

if __name__ == "__main__":
    create_audio_visual_split()
