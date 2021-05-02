import torch.utils.data
import tqdm


class SkywalkLeapDataset(torch.utils.data.Dataset):
    def __init__(self, skywalk_length, leapmotion_timestamps, leapmotion_data, skywalk_timestamps, skywalk_data):
        leapmotion_idx = 0
        self.skywalk_idx_to_leapmotion_map = []
        for skywalk_idx in tqdm.tqdm(range(skywalk_length, len(skywalk_timestamps))):
            while leapmotion_idx < len(leapmotion_timestamps) and \
                    leapmotion_timestamps[leapmotion_idx] < skywalk_timestamps[skywalk_idx]:
                leapmotion_idx += 1
            if leapmotion_idx == len(leapmotion_timestamps):
                break
            if leapmotion_timestamps[leapmotion_idx] - skywalk_timestamps[skywalk_idx] < 10 and \
                    skywalk_idx > skywalk_length:
                self.skywalk_idx_to_leapmotion_map += [(skywalk_idx, leapmotion_idx)]
        self.leapmotion_data = leapmotion_data
        self.skywalk_data = skywalk_data
        self.leapmotion_timestamps = leapmotion_timestamps
        self.skywalk_timestamps = skywalk_timestamps
        self.skywalk_length = skywalk_length

    def __len__(self):
        return len(self.skywalk_idx_to_leapmotion_map)

    def __getitem__(self, item):
        skywalk_idx, leapmotion_idx = self.skywalk_idx_to_leapmotion_map[item]
        skywalk_input = self.skywalk_data[skywalk_idx - self.skywalk_length: skywalk_idx]
        leapmotion_output = self.leapmotion_data[leapmotion_idx]
        return skywalk_input, leapmotion_output
