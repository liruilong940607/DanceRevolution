import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from dataset import DanceDataset, paired_collate_fn
from utils.functional import str2bool, load_data, load_data_aist
from generator import Generator
from PIL import Image
from keypoint2img import read_keypoints
from multiprocessing import Pool
from functools import partial
import time


import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


def rotmat2aa(rotmats):
    """
    Convert rotation matrices to angle-axis using opencv's Rodrigues formula.
    Args:
        rotmats: A np array of shape (..., 3, 3)
    Returns:
        A np array of shape (..., 3)
    """
    assert rotmats.shape[-1] == 3 and rotmats.shape[-2] == 3 and len(rotmats.shape) >= 3, 'invalid input dimension'
    orig_shape = rotmats.shape[:-2]
    rots = np.reshape(rotmats, [-1, 3, 3])
    r = R.from_dcm(rots)  # from_matrix
    aas = r.as_rotvec()
    return np.reshape(aas, orig_shape + (3,))


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).
    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest


pose_keypoints_num = 25
face_keypoints_num = 70
hand_left_keypoints_num = 21
hand_right_keypoints_num = 21


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='data/test')
    parser.add_argument('--data_type', type=str, default='2D', help='the type of pose data')
    parser.add_argument('--model', type=str, metavar='PATH', default='checkpoints/epoch_3000.pt')
    parser.add_argument('--json_dir', metavar='PATH', default='outputs/',
                        help='the generated pose data of json format')
    parser.add_argument('--image_dir', type=str, default='images',
                        help='the directory of visualization image')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--width', type=int, default=1280, help='the width pixels of image')
    parser.add_argument('--height', type=int, default=720, help='the height pixels of image')

    parser.add_argument('--seed', type=int, default=123,
                        help='random seed for data shuffling, dropout, etc.')
    parser.add_argument('--cuda', type=str2bool, nargs='?', metavar='BOOL',
                        const=True, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')

    parser.add_argument('--aist', action='store_true', help='test on AIST++')
    parser.add_argument('--rotmat', action='store_true', help='train rotation matrix')

    return parser.parse_args()


def visualize_json(fname_iter, image_dir, dance_name, dance_path, args):
    j, fname = fname_iter
    json_file = os.path.join(dance_path, fname)
    img = Image.fromarray(read_keypoints(json_file, (args.width, args.height),
                                         remove_face_labels=False, basic_point_only=False))
    img.save(os.path.join(f'{image_dir}/{dance_name}', f'frame{j:06d}.jpg'))


def visualize(args, worker_num=16):
    json_dir = args.json_dir
    image_dir = args.image_dir

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    dance_names = sorted(os.listdir(json_dir))
    for i, dance_name in enumerate(dance_names):
        dance_path = os.path.join(json_dir, dance_name)
        fnames = sorted(os.listdir(dance_path))
        if not os.path.exists(f'{image_dir}/{dance_name}'):
            os.makedirs(f'{image_dir}/{dance_name}')

        # Visualize json in parallel
        pool = Pool(worker_num)
        partial_func = partial(visualize_json, image_dir=image_dir,
                               dance_name=dance_name, dance_path=dance_path, args=args)
        pool.map(partial_func, enumerate(fnames))
        pool.close()
        pool.join()
        
        print(f'Visualize {dance_name}')


def write2json(dances, dance_names, args):
    assert len(dances) == len(dance_names),\
        "number of generated dance != number of dance_names"
    for i in range(len(dances)):
        num_poses = dances[i].shape[0]
        dances[i] = dances[i].reshape(num_poses, pose_keypoints_num, 2)
        dance_path = os.path.join(args.json_dir, dance_names[i])
        if not os.path.exists(dance_path):
            os.makedirs(dance_path)

        for j in range(num_poses):
            frame_dict = {'version': 1.2}
            # 2-D key points
            pose_keypoints_2d = []
            # Random values for the below key points
            face_keypoints_2d = []
            hand_left_keypoints_2d = []
            hand_right_keypoints_2d = []
            # 3-D key points
            pose_keypoints_3d = []
            face_keypoints_3d = []
            hand_left_keypoints_3d = []
            hand_right_keypoints_3d = []

            keypoints = dances[i][j]
            for k, keypoint in enumerate(keypoints):
                x = (keypoint[0] + 1) * 0.5 * args.width
                y = (keypoint[1] + 1) * 0.5 * args.height
                score = 0.8
                if k < pose_keypoints_num:
                    pose_keypoints_2d.extend([x, y, score])
                elif k < pose_keypoints_num + face_keypoints_num:
                    face_keypoints_2d.extend([x, y, score])
                elif k < pose_keypoints_num + face_keypoints_num + hand_left_keypoints_num:
                    hand_left_keypoints_2d.extend([x, y, score])
                else:
                    hand_right_keypoints_2d.extend([x, y, score])

            people_dicts = []
            people_dict = {'pose_keypoints_2d': pose_keypoints_2d,
                           'face_keypoints_2d': face_keypoints_2d,
                           'hand_left_keypoints_2d': hand_left_keypoints_2d,
                           'hand_right_keypoints_2d': hand_right_keypoints_2d,
                           'pose_keypoints_3d': pose_keypoints_3d,
                           'face_keypoints_3d': face_keypoints_3d,
                           'hand_left_keypoints_3d': hand_left_keypoints_3d,
                           'hand_right_keypoints_3d': hand_right_keypoints_3d}
            people_dicts.append(people_dict)
            frame_dict['people'] = people_dicts
            frame_json = json.dumps(frame_dict)
            with open(os.path.join(dance_path, f'frame{j:06d}_keypoints.json'), 'w') as f:
                f.write(frame_json)
        print(f'Writing json to -> {dance_path}')


def main():
    args = get_args()
    if args.aist:
        import vedo

    if not os.path.exists(args.json_dir):
        os.makedirs(args.json_dir)

    if args.aist:
        print ("test with AIST++ dataset!")
        music_data, dance_data, dance_names = load_data_aist(
            args.input_dir, interval=None, rotmat=args.rotmat)
    else:    
        music_data, dance_data, dance_names = load_data(
            args.input_dir, interval=None)

    device = torch.device('cuda' if args.cuda else 'cpu')

    test_loader = torch.utils.data.DataLoader(
        DanceDataset(music_data, dance_data),
        batch_size=args.batch_size,
        collate_fn=paired_collate_fn
    )

    generator = Generator(args.model, device)
    
    if args.aist and args.rotmat:
        from smplx import SMPL
        smpl = SMPL(model_path="/media/ruilongli/hd1/Data/smpl/", gender='MALE', batch_size=1)

    results = []
    random_id = 0  # np.random.randint(0, 1e4)
    for i, batch in enumerate(tqdm(test_loader, desc='Generating dance poses')):
        # Prepare data
        src_seq, src_pos, tgt_pose = map(lambda x: x.to(device), batch)
        pose_seq = generator.generate(src_seq[:, :1200], src_pos[:, :1200])  # first 20 secs
        results.append(pose_seq)

        if args.aist:
            np_dance = pose_seq[0].data.cpu().numpy()
            if args.rotmat:
                root = np_dance[:, :3]
                rotmat = np_dance[:, 3:].reshape([-1, 3, 3])
                rotmat = get_closest_rotmat(rotmat)
                smpl_poses = rotmat2aa(rotmat).reshape(-1, 24, 3)
                np_dance = smpl.forward(
                    global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
                    body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
                    transl=torch.from_numpy(root).float(),
                ).joints.detach().numpy()[:, 0:24, :]
            else:
                root = np_dance[:, :3]
                np_dance = np_dance + np.tile(root, (1, 24))
                np_dance[:, :3] = root
                np_dance = np_dance.reshape(np_dance.shape[0], -1, 3)
            print (np_dance.shape)
            # save
            save_path = os.path.join(args.json_dir, dance_names[i]+f"_{random_id:04d}")
            np.save(save_path, np_dance)
            # # visualize
            # for frame in np_dance:
            #     pts = vedo.Points(frame, r=20)
            #     vedo.show(pts, interactive=False)
            #     # time.sleep(0.02)
            # # exit()

    if args.aist:
        pass

    else:
        # Visualize generated dance poses
        np_dances = []
        for i in range(len(results)):
            np_dance = results[i][0].data.cpu().numpy()
            root = np_dance[:, 2*8:2*9]
            np_dance = np_dance + np.tile(root, (1, 25))
            np_dance[:, 2*8:2*9] = root
            np_dances.append(np_dance)
        write2json(np_dances, dance_names, args)
        visualize(args)


if __name__ == '__main__':
    main()
