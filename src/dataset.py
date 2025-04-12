import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils import normalize_data

class SequenceDataset(Dataset):
    def __init__(self, series, input_window, forecast_horizon):
        self.series = series
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.series) - self.input_window - self.forecast_horizon

    def __getitem__(self, idx):
        x = self.series[idx:idx + self.input_window]
        y = self.series[idx + self.input_window:idx + self.input_window + self.forecast_horizon]
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y)


class TimeSeriesDataset:
    def __init__(self, folder_path, input_window=30, forecast_horizon=1, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, normalize=True):
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.series = self._load_data_from_folder(folder_path)
        self.train_data, self.val_data, self.test_data, self.stats = self.split_time_series(
            self.series, train_ratio, val_ratio, test_ratio, normalize
        )

    def _load_data_from_folder(self, folder_path):
        data_frames = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path, sep=',')
                df.columns = df.columns.str.strip()
                df['start_meetperiode'] = pd.to_datetime(df['start_meetperiode'])
                df.sort_values('start_meetperiode', inplace=True)
                df['gem_intensiteit'] = pd.to_numeric(df['gem_intensiteit'], errors='coerce')
                df.dropna(subset=['gem_intensiteit'], inplace=True)
                data_frames.append(df[['start_meetperiode', 'gem_intensiteit']])
        full_data = pd.concat(data_frames, axis=0)
        return full_data['gem_intensiteit'].values.astype(np.float32)

    def split_time_series(self, series, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, normalize=True):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        if normalize:
            series, stats = normalize_data(series)
        else:
            stats = {'mean': None, 'std': None}
        n = len(series)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        return series[:train_end], series[train_end:val_end], series[val_end:], stats

    def get_split_dataset(self, split="train"):
        data = {
            "train": self.train_data,
            "val": self.val_data,
            "test": self.test_data
        }[split]
        return SequenceDataset(data, self.input_window, self.forecast_horizon)
