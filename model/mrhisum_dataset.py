import h5py
import numpy as np
import json
import torch
import h5py
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from networks.CrossAttentional.cam import CAM
from networks.sl_module.BottleneckTransformer import BottleneckTransformer

class MrHiSumDataset(Dataset):

    def __init__(self, mode, path):
        self.mode = mode
        self.dataset_path = 'dataset/mr_hisum1.h5'
        self.feature_file_path = 'dataset/mr_hisum_audio.h5'
        #self.dataset = 'dataset/tvsum/tvsum.h5'
        #self.feature_file = 'dataset/tvsum/tvsum.h5'
        self.split_file = path
        #self.split_file = 'dataset/Books & Literature_metadata_split.json'
        #self.split_file = 'dataset/Computers & Electronics_metadata_split.json' 
        #self.split_file = 'dataset/Beauty & Fitness_metadata_split.json' 
        #self.split_file = 'dataset/Arts & Entertainment_metadata_split.json' 
        #self.split_file = 'dataset/Games_metadata_split.json' 
        #self.split_file = 'dataset/Business & Industrial_metadata_split.json' 
        #self.split_file = 'dataset/News_metadata_split.json' 
        #self.split_file = 'dataset/Travel_metadata_split.json' 
        #self.split_file = 'dataset/Autos & Vehicles_metadata_split.json' 
        #self.split_file = 'dataset/sport_metadata_split.json' 
        
        # Don't open the HDF5 files in __init__, as they won't be properly shared across processes
        # Instead, we'll open them in __getitem__ when needed
        
        # Read the split data upfront since it's smaller and won't cause issues
        with open(self.split_file, 'r') as f:
            self.data = json.loads(f.read())

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.data[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """
        video_name = self.data[self.mode + '_keys'][index]
        d = {}
        d['video_name'] = video_name
        #concat
        visual_features = torch.Tensor(np.array(self.features_data[video_name + '/features']))  # Visual features
        audio_features = torch.Tensor(np.array(self.features_data[video_name + '/audio']))      # Audio features
        
        # 确保两个特征的时间步数（num_frames）一致
        assert visual_features.shape[0] == audio_features.shape[0], "Mismatch in time steps between visual and audio features."
        
        # 进行特征拼接
        # 拼接维度为 1（特征维度），时间维度保持不变
        #print(visual_features.shape,audio_features.shape)
        concat_features = torch.cat((visual_features, audio_features), dim=1)
        #print(concat_features.shape)
        # 存储拼接后的特征
        #d['features'] = concat_features

        d['features'] = torch.Tensor(np.array(self.features_data[video_name + '/features'])).detach() #visual
        #d['features'] = torch.Tensor(np.array(self.features_data[video_name + '/audio'])) #audio
        d['gtscore'] = torch.Tensor(np.array(self.video_data[video_name + '/gtscore'])).detach()
        d['audio'] = torch.Tensor(np.array(self.features_data[video_name + '/audio'])).detach()
        Cam = CAM()
        #d['multi'] = Cam(d['audio'], d['features']).detach()
        d['multi'] = concat_features.detach()
        if self.mode != 'train':
            #n_frames = d['features'].shape[0]
            n_frames = d['multi'].shape[0]
            cps = np.array(self.video_data[video_name + '/change_points'])
            d['n_frames'] = np.array(n_frames)
            d['picks'] = np.array([i for i in range(n_frames)])
            d['change_points'] = cps
            d['n_frame_per_seg'] = np.array([cp[1]-cp[0] for cp in cps])
            d['gt_summary'] = np.expand_dims(np.array(self.video_data[video_name + '/gt_summary']), axis=0)
        #print(d['features'])
        return d 
        
class MrHiSumDataset_tvsum(Dataset):

    def __init__(self, mode, path):
        self.mode = mode
        self.dataset = 'dataset/tvsum/tvsum.h5'
        self.feature_file = 'dataset/tvsum/tvsum.h5'
        #self.dataset = 'dataset/tvsum/tvsum.h5'
        #self.feature_file = 'dataset/tvsum/tvsum.h5'
        self.split_file = path
        #self.split_file = 'dataset/Books & Literature_metadata_split.json'
        #self.split_file = 'dataset/Computers & Electronics_metadata_split.json' 
        #self.split_file = 'dataset/Beauty & Fitness_metadata_split.json' 
        #self.split_file = 'dataset/Arts & Entertainment_metadata_split.json' 
        #self.split_file = 'dataset/Games_metadata_split.json' 
        #self.split_file = 'dataset/Business & Industrial_metadata_split.json' 
        #self.split_file = 'dataset/News_metadata_split.json' 
        #self.split_file = 'dataset/Travel_metadata_split.json' 
        #self.split_file = 'dataset/Autos & Vehicles_metadata_split.json' 
        #self.split_file = 'dataset/sport_metadata_split.json' 
        self.features_data = h5py.File(self.feature_file, 'r')
        self.video_data = h5py.File(self.dataset, 'r')

        with open(self.split_file, 'r') as f:
            self.data = json.loads(f.read())

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.data[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """
        video_name = self.data[self.mode + '_keys'][index]
        d = {}
        d['video_name'] = video_name
        #concat
        visual_features = torch.Tensor(np.array(self.features_data[video_name + '/features']))  # Visual features
        audio_features = torch.Tensor(np.array(self.features_data[video_name + '/audio']))      # Audio features
        
        # 确保两个特征的时间步数（num_frames）一致
        assert visual_features.shape[0] == audio_features.shape[0], "Mismatch in time steps between visual and audio features."
        
        # 进行特征拼接
        # 拼接维度为 1（特征维度），时间维度保持不变
        #print(visual_features.shape,audio_features.shape)
        concat_features = torch.cat((visual_features, audio_features), dim=1)
        #print(concat_features.shape)
        # 存储拼接后的特征
        #d['features'] = concat_features

        d['features'] = torch.Tensor(np.array(self.features_data[video_name + '/features'])).detach() #visual
        #d['features'] = torch.Tensor(np.array(self.features_data[video_name + '/audio'])) #audio
        d['gtscore'] = torch.Tensor(np.array(self.video_data[video_name + '/gtscore'])).detach()
        d['audio'] = torch.Tensor(np.array(self.features_data[video_name + '/audio'])).detach()
        Cam = CAM()
        #d['multi'] = Cam(d['audio'], d['features']).detach()
        d['multi'] = concat_features.detach()
        if self.mode != 'train':
            #n_frames = d['features'].shape[0]
            n_frames = d['multi'].shape[0]
            cps = np.array(self.video_data[video_name + '/change_points'])
            d['n_frames'] = np.array(n_frames)
            d['picks'] = np.array([i for i in range(n_frames)])
            d['change_points'] = cps
            d['n_frame_per_seg'] = np.array([cp[1]-cp[0] for cp in cps])
            d['gt_summary'] = np.expand_dims(np.array(self.video_data[video_name + '/gt_summary']), axis=0)
        #print(d['features'])
        return d 
'''class MrHiSumDataset(Dataset):
    def __init__(self, mode, window_size=30, overlap=15):
        self.mode = mode
        self.dataset = 'dataset/mr_hisum1.h5'
        self.feature_file = 'dataset/mr_hisum_audio.h5'
        self.split_file = 'dataset/mr_hisum_split.json'

        self.features_data = h5py.File(self.feature_file, 'r')
        self.video_data = h5py.File(self.dataset, 'r')

        with open(self.split_file, 'r') as f:
            self.data = json.loads(f.read())

        self.window_size = window_size  # 滑動窗口大小
        self.overlap = overlap  # 窗口重疊大小

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.data[self.mode + '_keys'])
        return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset. """
        video_name = self.data[self.mode + '_keys'][index]
        d = {}
        d['video_name'] = video_name
        visual_features = torch.Tensor(np.array(self.features_data[video_name + '/features']))  # Visual features
        audio_features = torch.Tensor(np.array(self.features_data[video_name + '/audio']))      # Audio features
        # 确保两个特征的时间步数（num_frames）一致
        assert visual_features.shape[0] == audio_features.shape[0], "Mismatch in time steps between visual and audio features."
        
        # 进行特征拼接
        # 拼接维度为 1（特征维度），时间维度保持不变
        #print(visual_features.shape,audio_features.shape)
        concat_features = torch.cat((visual_features, audio_features), dim=1)
        #print(concat_features.shape)
        # 存储拼接后的特征
        #d['features'] = concat_features
        # 原始的完整特徵
        #features = torch.Tensor(np.array(self.features_data[video_name + '/features']))  # visual
        #features = torch.Tensor(np.array(self.features_data[video_name + '/audio']))  # audio
        gtscore = torch.Tensor(np.array(self.video_data[video_name + '/gtscore']))
        audio = torch.Tensor(np.array(self.features_data[video_name + '/audio']))

        # 使用滑動窗口處理特徵和分數
        #features, gtscore, audio = self._apply_sliding_window(features, gtscore, audio)
        features, gtscore, audio = self._apply_sliding_window(concat_features, gtscore, audio)
        d['features'] = features
        d['gtscore'] = gtscore
        d['audio'] = audio

        if self.mode != 'train':
            n_frames = d['features'].shape[0]
            cps = np.array(self.video_data[video_name + '/change_points'])
            d['n_frames'] = np.array(n_frames)
            d['picks'] = np.array([i for i in range(n_frames)])
            d['change_points'] = cps
            d['n_frame_per_seg'] = np.array([cp[1] - cp[0] for cp in cps])
            d['gt_summary'] = np.expand_dims(np.array(self.video_data[video_name + '/gt_summary']), axis=0)

        return d

    def _apply_sliding_window(self, features, gtscore, audio):
        """ 使用滑動窗口處理特徵、分數和音頻特徵 """
        step_size = self.window_size - self.overlap
        feature_windows, gtscore_windows, audio_windows = [], [], []

        # 滑動窗口分段
        for start in range(0, len(features) - self.window_size + 1, step_size):
            feature_windows.append(features[start:start + self.window_size])
            gtscore_windows.append(gtscore[start:start + self.window_size])
            audio_windows.append(audio[start:start + self.window_size])

        # 將窗口展平並調整形狀
        feature_windows = torch.cat(feature_windows, dim=0)  # 展平為 (dim1 * dim2, dim3)
        gtscore_windows = torch.cat(gtscore_windows, dim=0)  # 展平為 (dim1 * dim2)
        audio_windows = torch.cat(audio_windows, dim=0)     # 展平為 (dim1 * dim2, dim3)

        return feature_windows, gtscore_windows, audio_windows
'''       
class BatchCollator(object):
    def __call__(self, batch):
        #video_name, features, gtscore, = [],[],[]
        video_name, features, gtscore, audio, multi= [],[],[],[], []
        # cps, nseg, n_frames, picks, gt_summary = [], [], [], [], []

        try:
            for data in batch:
                video_name.append(data['video_name'])
                features.append(data['features'].detach())
                gtscore.append(data['gtscore'].detach())
                audio.append(data['audio'].detach())
                multi.append(data['multi'].detach())
                # cps.append(data['change_points'])
                # nseg.append(data['n_frame_per_seg'])
                # n_frames.append(data['n_frames'])
                # picks.append(data['picks'])
                # gt_summary.append(data['gt_summary'])
        except Exception as e:
            import traceback
            print('Error in batch collator:', str(e))
            print(traceback.format_exc())
            raise  # Re-raise the exception to properly show the error
        #print(len(features))
        lengths_visual = torch.LongTensor(list(map(lambda x: x.shape[0], features)))
        max_len_visual = max(list(map(lambda x: x.shape[0], features)))

        mask_visual = torch.arange(max_len_visual)[None, :] < lengths_visual[:, None]
        
        lengths_audio = torch.LongTensor(list(map(lambda x: x.shape[0], audio)))
        max_len_audio = max(list(map(lambda x: x.shape[0], audio)))

        mask_audio = torch.arange(max_len_audio)[None, :] < lengths_audio[:, None]

        lengths_multi = torch.LongTensor(list(map(lambda x: x.shape[0], multi)))
        max_len_multi = max(list(map(lambda x: x.shape[0], multi)))

        mask_multi = torch.arange(max_len_multi)[None, :] < lengths_multi[:, None]
        
        frame_feat_visual = pad_sequence(features, batch_first=True)
        frame_feat_audio = pad_sequence(audio, batch_first=True)
        frame_feat_multi = pad_sequence(multi, batch_first=True)
        gtscore = pad_sequence(gtscore, batch_first=True)
        #audio = pad_sequence(audio, batch_first=True)
        #batch_data = {'video_name' : video_name, 'features' : frame_feat, 'gtscore':gtscore, 'mask':mask}
        batch_data = {'video_name' : video_name, 'features' : frame_feat_visual, 'audio':frame_feat_audio, 'multi':frame_feat_multi, 'gtscore':gtscore, 'mask':mask_visual, 'mask_audio':mask_audio, 'mask_multi':mask_multi}
        # batch_data = {'video_name' : video_name, 'features' : frame_feat, 'gtscore':gtscore, 'mask':mask, \
        #               'n_frames': n_frames, 'picks': picks, 'n_frame_per_seg': nseg, 'change_points': cps, \
        #                 'gt_summary': gt_summary}
        return batch_data