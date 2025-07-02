import h5py
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from networks.CrossAttentional.cam import CAM

class MrHiSumDataset(Dataset):
    """Dataset class for MrHiSum with process-safe HDF5 handling."""
    
    def __init__(self, mode, path):
        self.mode = mode
        self.dataset_path = 'dataset/mr_hisum1.h5'
        self.feature_file_path = 'dataset/mr_hisum_audio.h5'
        self.split_file = path
        
        # Only load the split file data in __init__, which is small and safe
        with open(self.split_file, 'r') as f:
            self.data = json.loads(f.read())
            
    def __len__(self):
        """Returns the number of items in the dataset."""
        self.len = len(self.data[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        """Get a single item from the dataset by index.
        
        Opens and closes HDF5 files for each access to prevent multiprocessing issues.
        """
        # Open HDF5 files only when needed and close them afterward
        with h5py.File(self.feature_file_path, 'r') as features_data, h5py.File(self.dataset_path, 'r') as video_data:
            try:
                video_name = self.data[self.mode + '_keys'][index]
                d = {}
                d['video_name'] = video_name
                
                # Load visual and audio features and detach them
                visual_features = torch.Tensor(np.array(features_data[video_name + '/features'])).detach()
                audio_features = torch.Tensor(np.array(features_data[video_name + '/audio'])).detach()
                
                # Ensure features have matching time steps
                assert visual_features.shape[0] == audio_features.shape[0], "Mismatch in time steps between visual and audio features."
                
                # Create concatenated features for multi-modal
                concat_features = torch.cat((visual_features, audio_features), dim=1)
                
                # Store all required features in the dictionary
                d['features'] = visual_features
                d['gtscore'] = torch.Tensor(np.array(video_data[video_name + '/gtscore'])).detach()
                d['audio'] = audio_features
                d['multi'] = concat_features.detach()
                cps = np.array(video_data[video_name + '/change_points'])
                d['change_points'] = cps
                # Additional data for validation and test modes
                if self.mode != 'train':
                    n_frames = d['multi'].shape[0]
                    #cps = np.array(video_data[video_name + '/change_points'])
                    d['n_frames'] = np.array(n_frames)
                    d['picks'] = np.array([i for i in range(n_frames)])
                    #d['change_points'] = cps
                    d['n_frame_per_seg'] = np.array([cp[1]-cp[0] for cp in cps])
                    d['gt_summary'] = np.expand_dims(np.array(video_data[video_name + '/gt_summary']), axis=0)
            
            except Exception as e:
                print(f"Error loading data for video {video_name}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                raise
                
        return d

# The original BatchCollator with improved error handling
class BatchCollator(object):
    def __call__(self, batch):
        video_name, features, gtscore, audio, multi, cps = [], [], [], [], [], []
        
        try:
            for data in batch:
                video_name.append(data['video_name'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])
                audio.append(data['audio'])
                multi.append(data['multi'])
                cps.append(data['change_points'])
        except Exception as e:
            import traceback
            print('Error in batch collator:', str(e))
            print(traceback.format_exc())
            raise  # Re-raise the exception to properly show the error
        
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
        
        batch_data = {
            'video_name': video_name, 
            'features': frame_feat_visual, 
            'audio': frame_feat_audio, 
            'multi': frame_feat_multi, 
            'gtscore': gtscore, 
            'mask': mask_visual,
            'mask_audio': mask_audio, 
            'mask_multi': mask_multi,
            'change_points': cps  # Keep as original list of NumPy arrays
        }
        
        return batch_data
