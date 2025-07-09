# import os
# import json
# from collections import defaultdict

# def check_duplicate_ids():
#     # Directory containing the JSON files
#     dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    
#     # List of category files to check (as provided in the user request)
#     categories = [
#         "Hobbies_Leisure_split.json",
#         "(Unknown)_split.json",
#         "Business_Industrial_split.json",
#         "Pets_Animals_split.json",
#         "Real_Estate_split.json",
#         "Shopping_split.json",
#         "News_split.json",
#         "Food_Drink_split.json",
#         "Home_Garden_split.json",
#         "Books_Literature_split.json",
#         "Beauty_Fitness_split.json",
#         "Travel_split.json",
#         "Health_split.json",
#         "Law_Government_split.json",
#         "Games_split.json",
#         "Computers_Electronics_split.json",
#         "Arts_Entertainment_split.json",
#         "Science_split.json",
#         "People_Society_split.json",
#         "Reference_split.json",
#         "Internet_Telecom_split.json",
#         "Autos_Vehicles_split.json",
#         "Sports_split.json",
#         "Jobs_Education_split.json",
#         "Finance_split.json",
#     ]
    
#     # Dictionary to store which categories each video ID appears in
#     video_id_to_categories = defaultdict(list)
    
#     # Total count of unique IDs across all categories
#     all_unique_ids = set()
    
#     # Count of IDs in each category
#     category_counts = {}
    
#     # Process each category file
#     for category_file in categories:
#         file_path = os.path.join(dataset_dir, category_file)
        
#         try:
#             with open(file_path, 'r') as f:
#                 data = json.load(f)
                
#             # Extract all video IDs from train, val, and test keys
#             video_ids = set()
#             for key_type in ['train_keys', 'val_keys', 'test_keys']:
#                 if key_type in data:
#                     video_ids.update(data[key_type])
            
#             # Store count of IDs in this category
#             category_name = category_file.replace('_split.json', '')
#             category_counts[category_name] = len(video_ids)
            
#             # Update the total unique IDs
#             all_unique_ids.update(video_ids)
            
#             # Record which categories each ID belongs to
#             for video_id in video_ids:
#                 video_id_to_categories[video_id].append(category_name)
                
#             print(f"Processed {category_file}: {len(video_ids)} IDs")
                
#         except FileNotFoundError:
#             print(f"Warning: File not found - {file_path}")
#             continue
#         except json.JSONDecodeError:
#             print(f"Warning: JSON decode error in file - {file_path}")
#             continue
    
#     # Find video IDs that appear in multiple categories
#     duplicate_ids = {
#         video_id: categories 
#         for video_id, categories in video_id_to_categories.items() 
#         if len(categories) > 1
#     }
    
#     # Print results
#     print("\n=== RESULTS ===")
#     if duplicate_ids:
#         print(f"\nFound {len(duplicate_ids)} video IDs that appear in multiple categories:")
#         for video_id, cats in sorted(duplicate_ids.items()):
#             print(f"{video_id} appears in: {', '.join(cats)}")
#     else:
#         print("\nNo duplicate video IDs found across categories.")
    
#     # Print summary statistics
#     print(f"\nTotal unique video IDs across all categories: {len(all_unique_ids)}")
#     print(f"Sum of IDs across all individual categories: {sum(category_counts.values())}")
    
#     # Print individual category counts
#     print("\nVideo count per category:")
#     for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
#         print(f"{category}: {count}")

# if __name__ == "__main__":
#     check_duplicate_ids()
import os
import json
from collections import defaultdict

def check_first_category_distribution():
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
    
    # 存儲每個視頻 ID 首次出現的分類
    video_first_category = {}
    
    # 各分類中的視頻 ID 集合
    original_category_ids = {}
    
    # 處理每個分類文件
    for category_file in categories:
        file_path = os.path.join(dataset_dir, category_file)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # 從 train、val 和 test keys 中提取所有視頻 ID
            video_ids = set()
            for key_type in ['train_keys', 'val_keys', 'test_keys']:
                if key_type in data:
                    video_ids.update(data[key_type])
            
            # 存儲該分類的原始 ID 集合
            category_name = category_file.replace('_split.json', '')
            original_category_ids[category_name] = video_ids
            
            # 記錄每個 ID 首次出現的分類
            for video_id in video_ids:
                if video_id not in video_first_category:
                    video_first_category[video_id] = category_name
                    
            print(f"處理 {category_file}: {len(video_ids)} 個 ID")
                
        except FileNotFoundError:
            print(f"警告: 找不到文件 - {file_path}")
            continue
        except json.JSONDecodeError:
            print(f"警告: JSON 解碼錯誤 - {file_path}")
            continue
    
    # 計算原始分類數量
    original_counts = {category: len(ids) for category, ids in original_category_ids.items()}
    
    # 計算新的分類分布（只將視頻分配給其首次出現的分類）
    new_category_counts = defaultdict(int)
    for video_id, category in video_first_category.items():
        new_category_counts[category] += 1
    
    # 計算變化
    changes = {}
    for category in original_counts.keys():
        original = original_counts.get(category, 0)
        new = new_category_counts.get(category, 0)
        diff = new - original
        percent = (diff / original * 100) if original > 0 else 0
        changes[category] = (original, new, diff, percent)
    
    # 打印結果
    print("\n=== 結果 ===")
    print("\n視頻總數（刪除重複後）:", len(video_first_category))
    print("原始所有分類中的視頻總數（含重複）:", sum(original_counts.values()))
    print("新分類中的視頻總數（應等於不重複總數）:", sum(new_category_counts.values()))
    
    # 打印新的分類分布
    print("\n如果僅將視頻歸類到首個分類的分布:")
    print(f"{'分類':25} {'原始數量':10} {'新數量':10} {'變化':10} {'百分比變化':10}")
    print("-" * 65)
    
    # 按新數量降序排序
    for category, (original, new, diff, percent) in sorted(
        changes.items(), key=lambda x: x[1][1], reverse=True):
        print(f"{category:25} {original:10d} {new:10d} {diff:+10d} {percent:+10.2f}%")

if __name__ == "__main__":
    check_first_category_distribution()
    