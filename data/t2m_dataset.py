from os.path import join as pjoin
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import random
import codecs as cs


# Original HumanML3D's collate_fn

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

"""
def collate_fn(batch):
    # Handles variable-length sequences dynamically.
    # Ensures text sequences are properly sorted for the Text Encoder's pack_padded_sequence requirement.

    is_eval = len(batch[0]) == 7
    
    if is_eval:
        # CRITICAL FIX: Sort by text caption length (sent_len is index 3) 
        # to satisfy pack_padded_sequence text encoder requirements
        batch.sort(key=lambda x: x[3], reverse=True)
        word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lengths, tokens = zip(*batch)
        
        # Safely convert lists of numpy arrays directly into torch tensors
        word_embeddings = torch.stack([torch.tensor(w, dtype=torch.float32) for w in word_embeddings])
        pos_one_hots = torch.stack([torch.tensor(p, dtype=torch.float32) for p in pos_one_hots])
        
        sent_lens = torch.tensor(sent_lens, dtype=torch.long)
        motions = torch.tensor(np.array(motions), dtype=torch.float32)
        m_lengths = torch.tensor(m_lengths, dtype=torch.long)
        
        return word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lengths, tokens
    else:
        # Training batch (sort by motion length, which is index 2)
        batch.sort(key=lambda x: x[2], reverse=True)
        captions, motions, m_lengths = zip(*batch)
        
        motions = torch.tensor(np.array(motions), dtype=torch.float32)
        m_lengths = torch.tensor(m_lengths, dtype=torch.long)
        
        return captions, motions, m_lengths
"""

class MotionDataset(data.Dataset):
    """
    Sliding window local-motion tracker dataset.
    """
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.data = []
        self.lengths = []
        id_list = []
        
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list, desc="Loading Motion Dataset"):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except Exception as e:
                print(e) # Some motion may not exist in the KIT-ML dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)
        self.mean = mean
        self.std = std
        print(f"Total number of motions {len(self.data)}, snippets {self.cumsum[-1]}")

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
            
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        motion = (motion - self.mean) / self.std
        return motion

class Text2MotionDatasetEval(data.Dataset):
    """
    Evaluation Text-to-Motion dataset with linguistic features extracted.
    """
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24

        data_dict = {}
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list, desc="Parsing Eval Texts"):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= self.max_motion_length):
                    continue
                text_data = []
                flag = False
                with open(pjoin(opt.text_dir, name + '.txt'), encoding='utf-8') as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= self.max_motion_length):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        # Sort names and lengths cleanly together
        sorted_pairs = sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        if sorted_pairs:
            name_list, length_list = zip(*sorted_pairs)
        else:
            name_list, length_list = (), ()

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print(f"Eval Pointer Pointing at {self.pointer}")
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        # Ensure uniform padding regardless of sentence length
        if len(tokens) < self.opt.max_text_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            # Pads short sentences to (max_text_len + 2)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # Truncate tokens so that with sos/eos it reaches EXACTLY max_text_len + 2
            tokens = tokens[:self.opt.max_text_len] 
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens) # This will naturally be max_text_len + 2
            
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Handle sub-sampling chunks based on dataset unit length configs
        coin2 = np.random.choice(['single', 'single', 'double']) if self.opt.unit_length < 10 else 'single'
        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            
        sub_idx = random.randint(0, len(motion) - m_length)
        motion = motion[sub_idx:sub_idx + m_length]
        motion = (motion - self.mean) / self.std

        # Uniform padding up to max execution window
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]))], axis=0)
            
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class Text2MotionDataset(data.Dataset):
    """
    Main Text-to-Motion Training Dataset wrapper.
    """
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24

        data_dict = {}
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list, desc="Parsing Train Texts"):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= self.max_motion_length):
                    continue
                text_data = []
                flag = False
                with open(pjoin(opt.text_dir, name + '.txt'), encoding='utf-8') as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= self.max_motion_length):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                pass

        # Maintain sorted pairings for sorted sequential operations or length filtering
        sorted_pairs = sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        if sorted_pairs:
            name_list, length_list = zip(*sorted_pairs)
        else:
            name_list, length_list = (), ()

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        
        text_data = random.choice(text_list)
        caption = text_data['caption']

        coin2 = np.random.choice(['single', 'single', 'double']) if self.opt.unit_length < 10 else 'single'
        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            
        sub_idx = random.randint(0, len(motion) - m_length)
        motion = motion[sub_idx:sub_idx + m_length]
        motion = (motion - self.mean) / self.std

        # Unified logic matching the evaluation configuration
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]))], axis=0)
            
        return caption, motion, m_length

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print(f"Train Pointer Pointing at {self.pointer}")