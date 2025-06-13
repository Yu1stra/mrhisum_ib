import os
import json
import math
import h5py
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from yt8m_reader import read_tfrecord
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf

# This code is created by referring to the code of https://github.com/google/youtube-8m

def parse_sequence_example(example_proto):
    context_features = {
        'id': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.VarLenFeature(tf.int64)
    }
    sequence_features = {
        'rgb': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        'audio': tf.io.FixedLenSequenceFeature([], dtype=tf.string)
    }

    return tf.io.parse_sequence_example(example_proto, context_features, sequence_features)

def dequantize(features):
    return features * 4.0 / 255.0 - 2.0

def read_tfrecord(dataset_path, file_name, random_id):
    yt8m_path = os.path.join(dataset_path, f"frame/{file_name}.tfrecord")
    dataset = tf.data.TFRecordDataset(yt8m_path)
    parsed_dataset = dataset.map(parse_sequence_example)
    for example in parsed_dataset:
        if random_id == example[0]['id'].numpy().decode('utf-8'):
            label_indices = example[0]['labels'].values
            features = tf.cast(tf.io.decode_raw(example[1]['rgb'], tf.uint8), tf.float32).numpy()
            audio =  tf.cast(tf.io.decode_raw(example[1]['audio'], tf.uint8), tf.float32).numpy()
            break
    
    return dequantize(features), label_indices.numpy().astype(int), dequantize(audio)
def align_most_replayed(file_name, random_id, youtube_id, duration):
    mostreplayed_json_path = f"dataset/most_replayed_{youtube_id}.json"
    
    aligned = []
    with open(mostreplayed_json_path, 'r') as fd:
        data = json.load(fd)
        mr_chunk_size = data[0]["durationMillis"]
        for n in range(int(duration)):
            bin_number = math.floor(n * 1000 / mr_chunk_size)
            if bin_number > 99.01:
                bin_number = 99
            
            aligned.append(data[bin_number]["intensityScoreNormalized"])
    
    return np.array(aligned)

def preprocess(dataset_path):
    meta_data = "dataset/metadata.csv"
    h5fd = h5py.File("dataset/mr_hisum_audio.h5", 'a')
    df = pd.read_csv(meta_data)
    
    for row in tqdm(df.itertuples()):
        feature, labels ,audio= read_tfrecord(dataset_path, row.yt8m_file, row.random_id)
        h5fd.create_dataset(f"{row.video_id}/features", data=feature)
        h5fd.create_dataset(f"{row.video_id}/audio", data=audio)
        #mostreplayed = align_most_replayed(row.yt8m_file, row.random_id, row.youtube_id, row.duration)
        #h5fd.create_dataset(f"{row.video_id}/gtscore", data=mostreplayed)
    
    h5fd.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help='the path where yt8m dataset exists')
    args = parser.parse_args()

    dataset_path = args.dataset_path

    preprocess(dataset_path)