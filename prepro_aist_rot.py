import os
import sys
import json
import random
import argparse
import essentia
import essentia.streaming
from essentia.standard import *
import librosa
import numpy as np
from extractor import FeatureExtractor

from aist_plusplus.loader import AISTDataset
from smplx import SMPL
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument('--input_audio_dir', type=str, default='/media/ruilongli/hd1/Data/aist_plusplus/wav')
parser.add_argument('--input_dance_dir', type=str, default='/media/ruilongli/hd1/Data/aist_plusplus/v1')
parser.add_argument('--smpl_dir', type=str, default='/media/ruilongli/hd1/Data/smpl')

parser.add_argument('--train_dir', type=str, default='data_aist_rot/train')
parser.add_argument('--test_dir', type=str, default='data_aist_rot/test')

parser.add_argument('--sampling_rate', type=int, default=30720) # 60fps
args = parser.parse_args()

extractor = FeatureExtractor()

if not os.path.exists(args.train_dir):
    os.mkdir(args.train_dir)
if not os.path.exists(args.test_dir):
    os.mkdir(args.test_dir)



def extract_acoustic_feature(input_audio_dir):
    print('---------- Extract features from raw audio ----------')
    musics = {}
    # onset_beats = []
    audio_fnames = sorted(os.listdir(input_audio_dir))
    # audio_fnames = audio_fnames[:20]  # for debug
    print(f'audio_fnames: {audio_fnames}')

    for audio_fname in tqdm(audio_fnames):
        audio_file = os.path.join(input_audio_dir, audio_fname)
        print(f'Process -> {audio_file}')
        ### load audio ###
        sr = args.sampling_rate
        # sr = 48000
        loader = essentia.standard.MonoLoader(filename=audio_file, sampleRate=sr)
        audio = loader()
        audio = np.array(audio).T

        melspe_db = extractor.get_melspectrogram(audio, sr)
        mfcc = extractor.get_mfcc(melspe_db)
        mfcc_delta = extractor.get_mfcc_delta(mfcc)
        # mfcc_delta2 = get_mfcc_delta2(mfcc)

        audio_harmonic, audio_percussive = extractor.get_hpss(audio)
        # harmonic_melspe_db = get_harmonic_melspe_db(audio_harmonic, sr)
        # percussive_melspe_db = get_percussive_melspe_db(audio_percussive, sr)
        chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, sr)
        # chroma_stft = extractor.get_chroma_stft(audio_harmonic, sr)

        onset_env = extractor.get_onset_strength(audio_percussive, sr)
        tempogram = extractor.get_tempogram(onset_env, sr)
        onset_beat = extractor.get_onset_beat(onset_env, sr)
        # onset_tempo, onset_beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        # onset_beats.append(onset_beat)

        onset_env = onset_env.reshape(1, -1)

        feature = np.concatenate([
            # melspe_db,
            mfcc,
            mfcc_delta,
            # mfcc_delta2,
            # harmonic_melspe_db,
            # percussive_melspe_db,
            # chroma_stft,
            chroma_cqt,
            onset_env,
            onset_beat,
            tempogram
        ], axis=0)

        feature = feature.transpose(1, 0)
        print(f'acoustic feature -> {feature.shape}')
        musics[audio_fname.replace(".wav", "")] = feature.tolist()
    

    return musics

def load_dance_data(dance_dir):
    print('---------- Loading pose keypoints ----------')
    aist_dataset = AISTDataset(dance_dir)
    seq_names = list(aist_dataset.mapping_seq2env.keys())
    print (seq_names)

    dances = {}

    for seq_name in tqdm(seq_names):
        print(f'Process -> {seq_name}')
        smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
            aist_dataset.motion_dir, seq_name)
        smpl_trans = smpl_trans / smpl_scaling
        nframes = smpl_poses.shape[0]
        njoints = 24

        r = R.from_rotvec(smpl_poses.reshape([nframes*njoints, 3])) 
        rotmat = r.as_dcm().reshape([nframes, njoints, 3, 3])

        dances[seq_name] = np.concatenate([
            smpl_trans,
            rotmat.reshape([nframes, njoints * 3 * 3])
        ], axis=-1).tolist()
        print(np.shape(dances[seq_name]))  # (nframes, 3 + 24 * 9)

    return dances 


def align(musics, dances):
    print('---------- Align the frames of music and dance ----------')
    # assert len(musics) == len(dances), \
    #     'the number of audios should be equal to that of videos'
    new_musics=[]
    new_dances=[]
    seq_names=[]
    for seq_name, dance in dances.items():
        audio_name = seq_name.split("_")[-2]
        music = musics[audio_name]
        min_seq_len = min(len(music), len(dance))
        print(f'music -> {np.array(music).shape}, ' +
              f'dance -> {np.array(dance).shape}, ' +
              f'min_seq_len -> {min_seq_len}')
        new_musics.append([music[j] for j in range(min_seq_len)])
        new_dances.append([dance[j] for j in range(min_seq_len)])
        seq_names.append(seq_name)
    return new_musics, new_dances, seq_names



def split_data(args, fnames):
    print('---------- Split data into train and test ----------')
    seq_names_ignore = [
        f.strip() for f in open(
            os.path.join(args.input_dance_dir, "ignore_list.txt"), "r"
            ).readlines()]
    seq_names_train = [
        f.strip() for f in open(
            os.path.join(args.input_dance_dir, "splits/crossmodal_train.txt"), "r"
            ).readlines()]
    seq_names_val = [
        f.strip() for f in open(
            os.path.join(args.input_dance_dir, "splits/crossmodal_val.txt"), "r"
            ).readlines()]
    seq_names_test = [
        f.strip() for f in open(
            os.path.join(args.input_dance_dir, "splits/crossmodal_test.txt"), "r"
            ).readlines()]
    seq_names_testval = seq_names_val + seq_names_test

    seq_names_train = [f for f in seq_names_train if f not in seq_names_ignore]
    seq_names_testval = [f for f in seq_names_testval if f not in seq_names_ignore]

    seq_names_train = [f for f in seq_names_train if f in fnames]
    seq_names_testval = [f for f in seq_names_testval if f in fnames]

    print (fnames)
    
    indices = list(range(len(fnames)))
    random.shuffle(indices)
    train = [fnames.index(f) for f in seq_names_train]
    test = [fnames.index(f) for f in seq_names_testval]
    print (len(train), len(test))
    random.shuffle(train)
    random.shuffle(test)

    return train, test


def save(args, musics, dances, seq_names, musics_raw):
    print('---------- Save to text file ----------')
    fnames = seq_names
    # fnames = fnames[:20]  # for debug
    assert len(fnames) == len(musics) == len(dances), 'alignment'

    train_idx, test_idx = split_data(args, fnames)
    train_idx = sorted(train_idx)
    print(f'train ids: {[fnames[idx] for idx in train_idx]}')
    test_idx = sorted(test_idx)
    print(f'test ids: {[fnames[idx] for idx in test_idx]}')

    print('---------- train data ----------')

    for idx in train_idx:
        with open(os.path.join(args.train_dir, f'{fnames[idx]}.json'), 'w') as f:
            sample_dict = {
                'id': fnames[idx],
                'music_array': musics[idx],
                'dance_array': dances[idx]
            }
            json.dump(sample_dict, f)

    print('---------- test data ----------')
    for idx in test_idx:
        audio_name = fnames[idx].split("_")[-2]
        with open(os.path.join(args.test_dir, f'{fnames[idx]}.json'), 'w') as f:
            sample_dict = {
                'id': fnames[idx],
                'music_array': musics_raw[audio_name], # musics[idx+i],
                'dance_array': dances[idx]
            }
            json.dump(sample_dict, f)



if __name__ == '__main__':

    musics_raw = extract_acoustic_feature(args.input_audio_dir)
    dances = load_dance_data(args.input_dance_dir)

    musics, dances, seq_names = align(musics_raw, dances)
    save(args, musics, dances, seq_names, musics_raw)
