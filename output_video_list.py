import torch
import pandas as pd
import glob

def main():
    meta_path = "/home/jay/MR.HiSum/dataset/metadata.csv"
    exist_file = glob.glob(f"/home/jay/usb/yt_video/*.mp4")
    exist = list(map(lambda x : (x.split(".")[0]).split("/")[-1], exist_file))
    print(f"exist len :{len(exist)}")
    #print(exist)
    df = pd.read_csv(meta_path)
    df = df["youtube_id"]
    output_path="/home/jay/usb/yt_video/video_list.txt"
    tmp=0
    tmp_n=0
    with open( output_path, 'a') as f:
        print(len(df))
        for i in df:
            for j in exist:
                if str(i) == str(j):
                    tmp+=1
                    print(f"https://www.youtube.com/watch?v={i} ,is not been write\n")
                    break
                elif j == exist[-1]:
                    tmp_n+=1
                    f.write(f"https://www.youtube.com/watch?v={i}\n")
                    print(f"https://www.youtube.com/watch?v={i} ,is been write\n")
    print(f"exist: {tmp}, not exist: {tmp_n}")


if __name__ == "__main__":
    main()