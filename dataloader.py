import h5py
import numpy as np

from dataset import SkywalkLeapDataset


def load_h5(filename: str):
    data_file = h5py.File(filename, 'r')

    leapmotion_timestamps, leapmotion_data, skywalk_timestamps, skywalk_data = \
        data_file["leapmotion_timestamps"], \
        data_file["leapmotion_data"], \
        data_file["skywalk_timestamps"], \
        data_file["skywalk_data"]

    leapmotion_timestamps, leapmotion_data, skywalk_timestamps, skywalk_data = np.array(
        leapmotion_timestamps), np.array(
        leapmotion_data), np.array(skywalk_timestamps), np.array(skywalk_data)

    skywalk_data = np.array(skywalk_data)
    for i in range(skywalk_data.shape[1]):
        skywalk_data[:, i] = skywalk_data[:, i] / 10000
        # skywalk_data[:, i] = normalize(skywalk_data[:, i])

    leapmotion_data = np.array(leapmotion_data)
    # leapmotion_data[:, 1] = (leapmotion_data[:, 1] > - math.pi / 4) * leapmotion_data[:, 1] - (leapmotion_data[:, 1] <= - math.pi / 4) * leapmotion_data[:, 1]
    # leapmotion_data[:, 3] = - leapmotion_data[:, 1]
    # leapmotion_data[:, 4] = - leapmotion_data[:, 2]
    # leapmotion_data[:, 1:5][leapmotion_data[:, 1:5] < 0] = 0
    # leapmotion_data[:, 3] = normalize(leapmotion_data[:, 3])
    # leapmotion_data[:, 4] = normalize(leapmotion_data[:, 4])
    leapmotion_data[:, 3] = (leapmotion_data[:, 3] - 60) / 20
    leapmotion_data[:, 4] = (leapmotion_data[:, 4] - 100) / 40
    print("dataset_length", len(skywalk_timestamps))

    train_val_div = len(skywalk_timestamps) // 4 * 3
    train_dataset = SkywalkLeapDataset(
        243,
        leapmotion_timestamps,
        leapmotion_data,
        skywalk_timestamps[200:train_val_div],
        skywalk_data[200:train_val_div]
    )

    val_dataset = SkywalkLeapDataset(
        243,
        leapmotion_timestamps,
        leapmotion_data,
        skywalk_timestamps[train_val_div:],
        skywalk_data[train_val_div:]
    )
    return train_dataset, val_dataset
