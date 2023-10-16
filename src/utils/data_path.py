import os
from itertools import product
import pandas as pd


class UBFC_RawDataPathLoader:
    def __init__(self, data_root_path) -> None:
        self.data_root_path = data_root_path
        self.dataset_root_path = self.data_root_path + 'UBFC'

        self.video_name = '/vid.avi'
        self.ppg_name = '/ground_truth.txt'
        self.video_fps = 30
        self.ppg_sampling_rate = 30.

        self.person_list = self.set_person_list()
        self.data_list = self.set_data_list()

    def set_person_list(self):
        person_list = []
        for person in os.listdir(self.dataset_root_path):
            if person.startswith('subject'):
                person_list.append(person[7:])
        return person_list
    
    def set_data_list(self):
        data_list = []
        for person in self.person_list:
            data_temp = f"/subject{person}"
            if os.path.isdir(self.dataset_root_path + data_temp):
                data_list.append(data_temp)
        return data_list


class PURE_RawDataPathLoader:

    def __init__(self, data_root_path) -> None:
        self.data_root_path = data_root_path
        self.dataset_root_path = self.data_root_path + 'PURE'

        self.video_name = '/png'
        self.ppg_name = '/json'
        self.video_fps = 35.14
        self.ppg_sampling_rate = 60

        self.person_list = self.set_person_list()
        self.task_list = self.set_task_list()
        self.data_list = self.set_data_list()

    def set_person_list(self):
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    def set_task_list(self):
        return [1, 2, 3, 4, 5, 6]
    
    def set_data_list(self):
        data_list = []
        for person, task in product(self.person_list, self.task_list):
            data_temp = f'/{person:02}-{task:02}'
            if os.path.isdir(self.dataset_root_path + data_temp):
                data_list.append(data_temp)
        return data_list


if __name__ == '__main__':

    for p in os.listdir("/mnt/e/UBFC_DATASET/UBFC"):
        print(p)
