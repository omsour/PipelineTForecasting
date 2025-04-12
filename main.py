# src/train.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.train import train_model



# Adjust this to point to your CSV folder.
data_folder = r"C:\Users\omarm\Desktop\Data Science\Project\PipelineTForecasting\data\GRT02_MORO_1387_2_data"
trained_model = train_model(data_folder=data_folder)