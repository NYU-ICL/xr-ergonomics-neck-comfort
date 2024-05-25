import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dataset import MCLDataset
from model import MCLNet
from scipy import signal
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='XR Ergonomics')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--num-sessions', type=int, default=9, help='Number of data collection sessions')
parser.add_argument('--num-channels', type=int, default=4, help='Number of sEMG channels')
parser.add_argument('--num-dofs', type=int, default=2, help='Head rotation degree of freedom')
parser.add_argument('--num-hidden-units', type=int, default=20, help='Number of hidden units in MCLNet')
parser.add_argument('--hop-length-test', type=int, default=4, help='Hop length for evaluation')
parser.add_argument('--hop-length-train', type=int, default=2, help='Hop length for training')
parser.add_argument('--num-input-samples', type=int, default=8, help='Input sequence length')
parser.add_argument('--num-output-samples', type=int, default=4, help='Output sequence length')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num-epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay factor')
parser.add_argument('--disable-augmentation', action='store_true', default=False, help='Disable left-right symmetry augmentation')
FLAGS = parser.parse_args()


def prepare_data(args):
    # Butterworth bandpass filter
    emg_bandpass_freq_low = 20
    emg_bandpass_freq_high = 150
    emg_butterworth_order = 4
    # RMS envelope filter
    emg_rms_window_length = 1000
    # Butterworth lowpass filter
    emg_lowpass_freq = 1
    # Mean filter
    angle_mean_window_length = 30
    velocity_mean_window_length = 10
    acceleration_mean_window_length = 30
    # Butterworth lowpass filter
    velocity_butterworth_order = 4
    velocity_lowpass_freq = 5
    # Resampling
    resampling_rate = 20
    mcl_train_all_users = []
    mcl_test_all_users = []
    angles_train_all_users = []
    angles_test_all_users = []
    accelerations_train_all_users = []
    accelerations_test_all_users = []
    for user in args.users:
        mcl_user = []
        angles_user = []
        accelerations_user = []
        folderpath = os.path.join(args.data_path, user)
        assert os.path.exists(folderpath), f"The data for user '{user}' not found!"
        for sid in range(args.num_sessions):
            emg_filepath = os.path.join(folderpath, f"emg_{sid:d}.csv")
            motion_filepath = os.path.join(folderpath, f"head_{sid:d}.csv")
            ##############
            #  EMG data  #
            ##############
            emg = np.loadtxt(emg_filepath, dtype=np.float32, delimiter='\t')
            motion = np.loadtxt(motion_filepath, dtype=np.float32, delimiter='\t')
            idle = max(emg[0, 0], motion[0, 0])
            emg = emg[:, :1 + args.num_channels]
            emg = emg[emg[:, 0] >= idle]
            timestamps = emg[:, 0]
            # Constant detrending
            emg[:, 1:] -= emg[:, 1:].mean(0)
            # Remove spikes
            emg[:, 1:3][emg[:, 1:3] > 0.2] = 0.2
            emg[:, 1:3][emg[:, 1:3] < -0.2] = -0.2
            emg[:, 3:5][emg[:, 3:5] > 0.1] = 0.1
            emg[:, 3:5][emg[:, 3:5] < -0.1] = -0.1
            # Butterworth bandpass filter
            emg_sampling_rate = len(emg) / (timestamps[-1] - timestamps[0])
            low = (2 * emg_bandpass_freq_low) / emg_sampling_rate
            high = (2 * emg_bandpass_freq_high) / emg_sampling_rate
            sos_bandpass = signal.butter(emg_butterworth_order, [low, high], btype='bandpass', analog=False, output='sos')
            emg[:, 1:] = signal.sosfilt(sos_bandpass, emg[:, 1:], axis=0)
            # RMS envelope filter
            rms_window = np.ones(emg_rms_window_length, dtype=np.float32) / emg_rms_window_length
            for i in range(1, 1 + args.num_channels):
                emg[:, i] = np.sqrt(signal.convolve(emg[:, i]**2, rms_window, mode='same', method='auto'))
            # Butterworth lowpass filter
            low = (2 * emg_lowpass_freq) / emg_sampling_rate
            sos_lowpass = signal.butter(emg_butterworth_order, low, btype='lowpass', analog=False, output='sos')
            emg[:, 1:] = signal.sosfilt(sos_lowpass, emg[:, 1:], axis=0)
            # Left/Right normalization
            emg[:, 1] *= (emg[:, 2].max() / emg[:, 1].max())
            emg[:, 3] *= (emg[:, 4].max() / emg[:, 3].max())
            # Compute MCL
            mcl = emg[:, 1:].sum(1)
            #################
            #  Motion data  #
            #################
            motion = motion[motion[:, 0] >= idle]
            angle_timestamps = motion[:, 0]
            velocity_timestamps = angle_timestamps[1:-1]
            motion = motion[:, 1:1 + args.num_dofs]
            motion[motion >= 180] -= 360
            # Mean filter
            angle_mean_window = np.ones(angle_mean_window_length, dtype=np.float32) / angle_mean_window_length
            velocity_mean_window = np.ones(velocity_mean_window_length, dtype=np.float32) / velocity_mean_window_length
            acceleration_mean_window = np.ones(acceleration_mean_window_length, dtype=np.float32) / acceleration_mean_window_length
            # Butterworth lowpass filter
            velocity_sampling_rate = len(velocity_timestamps) / (velocity_timestamps[-1] - velocity_timestamps[0])
            velocity_low = (2 * velocity_lowpass_freq) / velocity_sampling_rate
            velocity_sos_lowpass = signal.butter(velocity_butterworth_order, velocity_low, btype='lowpass', analog=False, output='sos')
            # Angles
            angles = motion.copy()
            for i in range(args.num_dofs):
                angles[:, i] = signal.convolve(angles[:, i], angle_mean_window, mode='same', method='auto')
            # Velocities
            velocities = ((motion[2:] - motion[:-2]).transpose() / (angle_timestamps[2:] - angle_timestamps[:-2])).transpose()
            for i in range(args.num_dofs):
                velocities[:, i] = signal.convolve(velocities[:, i], velocity_mean_window, mode='same', method='auto')
            velocities = signal.sosfilt(velocity_sos_lowpass, velocities, axis=0)
            # Accelerations
            accelerations = ((velocities[2:] - velocities[:-2]).transpose() / (velocity_timestamps[2:] - velocity_timestamps[:-2])).transpose()
            for i in range(args.num_dofs):
                accelerations[:, i] = signal.convolve(accelerations[:, i], acceleration_mean_window, mode='same', method='auto')
            ################
            #  Resampling  #
            ################
            num_samples = int(resampling_rate * (timestamps[-1] - idle))
            mcl = signal.resample(mcl, num_samples, t=None, axis=0, window=None, domain='time')[resampling_rate:-resampling_rate]
            angles = signal.resample(angles, num_samples, t=None, axis=0, window=None, domain='time')[resampling_rate:-resampling_rate]
            accelerations = signal.resample(accelerations, num_samples, t=None, axis=0, window=None, domain='time')[resampling_rate:-resampling_rate]
            # Gather results
            mcl_user.append(mcl.astype(np.float32))
            angles_user.append(angles.astype(np.float32))
            accelerations_user.append(accelerations.astype(np.float32))
        # Cross-user normalization
        mcl_min = np.concatenate(mcl_user, axis=0).min()
        for sid in range(len(mcl_user)):
            mcl_user[sid] -= mcl_min
        mcl_max = np.concatenate(mcl_user, axis=0).max()
        for sid in range(len(mcl_user)):
            mcl_user[sid] /= mcl_max
        # Build dataset
        for sid in range(len(mcl_user)):
            mcl = mcl_user[sid]
            angles = angles_user[sid]
            accelerations = accelerations_user[sid]
            if sid < args.num_sessions - 2:
                for t in range(0, len(mcl) - args.num_input_samples + 1, args.hop_length_train):
                    mcl_train_all_users.append(mcl[t:t + args.num_input_samples])
                    angles_train_all_users.append(angles[t:t + args.num_input_samples])
                    accelerations_train_all_users.append(accelerations[t:t + args.num_input_samples])
                    if not args.disable_augmentation:
                        mcl_train_all_users.append(mcl[t:t + args.num_input_samples])
                        angles_chunk = angles[t:t + args.num_input_samples].copy()
                        angles_chunk[:, 1] *= -1
                        angles_train_all_users.append(angles_chunk)
                        accelerations_chunk = accelerations[t:t + args.num_input_samples].copy()
                        accelerations_chunk[:, 1] *= -1
                        accelerations_train_all_users.append(accelerations_chunk)
            else:
                for t in range(0, len(mcl) - args.num_input_samples + 1, args.hop_length_test):
                    mcl_test_all_users.append(mcl[t:t + args.num_input_samples])
                    angles_test_all_users.append(angles[t:t + args.num_input_samples])
                    accelerations_test_all_users.append(accelerations[t:t + args.num_input_samples])
        print(f"Data preparation for user '{user}' completed")
    mcl_train_all_users = np.stack(mcl_train_all_users, axis=0)
    angles_train_all_users = np.stack(angles_train_all_users, axis=0)
    accelerations_train_all_users = np.stack(accelerations_train_all_users, axis=0)
    mcl_test_all_users = np.stack(mcl_test_all_users, axis=0)
    angles_test_all_users = np.stack(angles_test_all_users, axis=0)
    accelerations_test_all_users = np.stack(accelerations_test_all_users, axis=0)
    angles_max = np.maximum(np.abs(angles_train_all_users).max((0, 1)), np.abs(angles_test_all_users).max((0, 1)))
    angles_train_all_users = (angles_train_all_users / angles_max).transpose(0, 2, 1)
    angles_test_all_users = (angles_test_all_users / angles_max).transpose(0, 2, 1)
    accelerations_max = np.maximum(np.abs(accelerations_train_all_users).max((0, 1)), np.abs(accelerations_test_all_users).max((0, 1)))
    accelerations_train_all_users = (accelerations_train_all_users / accelerations_max).transpose(0, 2, 1)
    accelerations_test_all_users = (accelerations_test_all_users / accelerations_max).transpose(0, 2, 1)
    # print("Max angles: {np.array2string(angles_max, precision=2, separator=' ')} | Max accelerations: {np.array2string(accelerations_max, precision=2, separator=' ')}")
    print("Training MCL shape:", mcl_train_all_users.shape)
    print("Training angles shape:", angles_train_all_users.shape)
    print("Training accelerations shape:", accelerations_train_all_users.shape)
    print("Testing MCL shape:", mcl_test_all_users.shape)
    print("Testing angles shape:", angles_test_all_users.shape)
    print("Testing accelerations shape:", accelerations_test_all_users.shape)
    np.save(os.path.join(args.dataset_path, "mcl_train"), mcl_train_all_users)
    np.save(os.path.join(args.dataset_path, "angle_train"), angles_train_all_users)
    np.save(os.path.join(args.dataset_path, "acc_train"), accelerations_train_all_users)
    np.save(os.path.join(args.dataset_path, "mcl_test"), mcl_test_all_users)
    np.save(os.path.join(args.dataset_path, "angle_test"), angles_test_all_users)
    np.save(os.path.join(args.dataset_path, "acc_test"), accelerations_test_all_users)


def evaluate(model, dataloader, args):
    model.eval()
    errors = []
    losses_regression = []
    with torch.no_grad():
        for mcl, angles, accelerations in dataloader:
            if args.cuda:
                mcl, angles, accelerations = mcl.cuda(), angles.cuda(), accelerations.cuda()
            drop_size = args.num_input_samples - args.num_output_samples
            mcl = mcl[..., drop_size // 2:-(drop_size - drop_size // 2)]
            preds = model(angles, accelerations)
            loss_regression = F.mse_loss(preds, mcl)
            losses_regression.append(loss_regression.item())
            error = torch.abs(preds - mcl).mean()
            errors.append(error.item())
    loss_regression = np.mean(losses_regression)
    error = np.mean(errors)
    return loss_regression, error


def train_epoch(model, dataloader, optimizer, args):
    model.train()
    errors = []
    losses_regression = []
    for mcl, angles, accelerations in dataloader:
        if args.cuda:
            mcl, angles, accelerations = mcl.cuda(), angles.cuda(), accelerations.cuda()
        drop_size = args.num_input_samples - args.num_output_samples
        mcl = mcl[..., drop_size // 2:-(drop_size - drop_size // 2)]
        preds = model(angles, accelerations)
        loss_regression = F.mse_loss(preds, mcl)
        losses_regression.append(loss_regression.item())
        error = torch.abs(preds - mcl).mean()
        errors.append(error.item())
        optimizer.zero_grad()
        loss_regression.backward()
        optimizer.step()
    loss_regression = np.mean(losses_regression)
    error = np.mean(errors)
    return loss_regression, error


def train(args):
    data_ready = 1
    data_ready *= os.path.isfile(os.path.join(args.dataset_path, "mcl_train.npy"))
    data_ready *= os.path.isfile(os.path.join(args.dataset_path, "angle_train.npy"))
    data_ready *= os.path.isfile(os.path.join(args.dataset_path, "acc_train.npy"))
    data_ready *= os.path.isfile(os.path.join(args.dataset_path, "mcl_test.npy"))
    data_ready *= os.path.isfile(os.path.join(args.dataset_path, "angle_test.npy"))
    data_ready *= os.path.isfile(os.path.join(args.dataset_path, "acc_test.npy"))
    if not data_ready:
        prepare_data(args)
    model = MCLNet(args)
    if args.cuda:
        model.cuda()
    trainloader = load_data(args, True)
    testloader = load_data(args, False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        if epoch in [10]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        loss_test, error_test = evaluate(model, testloader, args)
        loss_train, error_train = train_epoch(model, trainloader, optimizer, args)
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch:d} | Time: {elapsed_time/60:.2f} min | Train loss: {loss_train:.3f} | Test loss: {loss_test:.3f} | Train error: {error_train:.3f} | Test error: {error_test:.3f}")
        info = {'epoch': epoch, 'state_dict': model.state_dict()}
        filepath = os.path.join(args.model_path, f"checkpoint-{epoch:d}.ckpt")
        torch.save(info, filepath)


def load_data(args, train=True):
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    dataset = MCLDataset(args.dataset_path, train)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train, **kwargs)


def main(args):
    FLAGS.users = ['A', 'B', 'C', 'D', 'E', 'F']
    args.cuda = torch.cuda.is_available()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    args.data_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'Data')
    args.dataset_path = os.path.join(os.getcwd(), 'Dataset')
    args.model_path = os.path.join(os.getcwd(), 'Checkpoints')
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)
    for user in args.users:
        assert os.path.exists(os.path.join(args.data_path, user)), f"The data for user {user} not found!"
    train(args)


if __name__ == '__main__':
    main(FLAGS)
