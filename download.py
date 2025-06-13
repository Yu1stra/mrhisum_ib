import os
import pandas as pd
from yt_dlp import YoutubeDL

def download_videos_from_csv(csv_path, output_path="ytaudio", size_limit_gb=350, num=1):
    fail_list=0
    failed_path = os.path.join(output_path, "failed_downloads.txt")
    # 讀取 CSV 檔案
    df = pd.read_csv(csv_path)
    
    # 確保輸出目錄存在
    os.makedirs(output_path, exist_ok=True)
    
    # 檢查 CSV 是否包含必要欄位
    if 'random_id' not in df.columns or 'youtube_id' not in df.columns:
        raise ValueError("CSV 必須包含 'random_id' 和 'youtube_id' 欄位")
    
    # 紀錄總下載大小（位元組）與失敗的下載
    total_bytes_downloaded = 0
    failed_downloads = []
    total_gb = 0
    # 回呼函數來追蹤下載進度
    def progress_hook(d):
        nonlocal total_bytes_downloaded
        if d['status'] == 'downloading':
            downloaded_bytes = d.get('downloaded_bytes', 0)
            total_bytes_downloaded += downloaded_bytes
            print(f"已下載大小：{total_bytes_downloaded / (1024**3):.2f} GB", end="\r")
        elif d['status'] == 'finished':
            print("\n下載完成！")
    
    # 下載每個視頻
    for _, row in df.iterrows():
        print("file_num=",num)
        print(f"總共下載大小：{total_bytes_downloaded / (1024**3):.2f} GB")
        random_id = row['random_id']
        youtube_id = row['youtube_id']
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        output_file = os.path.join(output_path, f"{random_id}.flac")

        if os.path.exists(output_file):
            print(f"檔案已存在，跳過下載：{output_file}")
            exist_list+=1
            continue
            
        # 停止下載如果超過限制
        if total_bytes_downloaded / (1024**3) > size_limit_gb:
            print(f"已達下載上限 {size_limit_gb} GB，停止下載")
            break
        
        # 自定義檔案名稱並確保包含聲音與影像
        """ydl_opts = {
            'outtmpl': os.path.join(output_path, f"{random_id}.%(ext)s"),
            'format': 'bestvideo+bestaudio/best',  # 確保同時下載聲音與影像
            'merge_output_format': 'mp4',         # 合併後輸出為 MP4 格式
            'progress_hooks': [progress_hook],   # 註冊回呼函數
        }"""
        ydl_opts = {
            'cookies-from-browser' :'chrome',
            'outtmpl': os.path.join(output_path, f"{random_id}.%(ext)s"),
            'format': 'bestaudio/best',  # 僅下載最佳音訊
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'flac',  # 儲存為無損音訊格式 FLAC
            }],
            'progress_hooks': [progress_hook],  # 註冊回呼函數
            'keepvideo': False  # 僅保留音訊
        }
        
        print(f"正在下載：{url} -> 檔名：{random_id}")
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            
            print(f"下載失敗：{youtube_id}，錯誤：{e}")
            with open(failed_path, "a") as f:
                f.write(f"下載失敗：{youtube_id}，錯誤：{e}\n")
            print(f"下載失敗的影片已記錄於：{failed_path}")
            fail_list+=1
        with open(failed_path, "a") as f:
            f.write(f"n={num},total_gb = {total_bytes_downloaded / (1024**3)}\n")
        num+=1
        
    
    # 轉換總下載大小為 GB
    total_gb = total_bytes_downloaded / (1024**3)
    print(f"總共下載大小：{total_gb:.2f} GB")
    return total_gb,fail_list

# 範例使用
if __name__ == "__main__":
    csv_path = "dataset/metadata.csv"
    total_gb,fail_list=download_videos_from_csv(csv_path)
    print("exist: ",fail_list)
