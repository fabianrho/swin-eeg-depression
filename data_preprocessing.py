import numpy as np
import mne
from tqdm import tqdm
import glob
import shutil
import os
import random

split = False



# timeframe_seconds = 1
# overlap_percentage = 0.5


for timeframe_seconds in [4,6]:
    for overlap_percentage in [0, 0.25, 0.5,0.75]:

        max_samples_per_patient = 10000000000000000

        def clean_channel_names(data):
            old_names = data.info['ch_names']
            new_names = data.info['ch_names']
            new_dict = {}
            # add the new names to the dictionary
            for i in range(0, len(old_names)):
                new_dict[old_names[i]] = new_names[i].replace('EEG ', '').replace('-LE', '')
            new_names = data.info['ch_names'][1:]

            data.info.rename_channels(new_dict)

            return data

        def load_data(path, filter = True):
            data = mne.io.read_raw_edf(path, verbose = False, preload=True)

            data = clean_channel_names(data)


            if "23A-23R" in data.info['ch_names']:
                data=data.drop_channels("23A-23R")
            if "24A-24R" in data.info['ch_names']:
                data=data.drop_channels("24A-24R")
            if "A2-A1" in data.info['ch_names']:
                data=data.drop_channels("A2-A1")

            montage = mne.channels.make_standard_montage("standard_1020")
            data = data.set_montage(montage, verbose = "CRITICAL")

            if filter:
                # Filter settings
                low_cut = 0.1
                hi_cut  = 70

                data = data.filter(low_cut, hi_cut, verbose = "CRITICAL")

                # notch filter
                data = data.notch_filter(50, verbose = "CRITICAL")

            data.set_eeg_reference(ref_channels="average", verbose = "CRITICAL")
            

            return data



        save_folder = f"data/data_{timeframe_seconds}s_{overlap_percentage}overlap"


        files = os.listdir('data_mdd')

        healthy_files_EC = [file for file in files if file.split(" ")[0] == "H" and "EC" in file.split(" ")[-1]]
        mdd_files_EC = [file for file in files if file.split(" ")[0] == "MDD" and "EC" in file.split(" ")[-1]]
        files = healthy_files_EC + mdd_files_EC

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for file in tqdm(files):
            data = load_data(f'data_mdd/{file}')
            timepoints = timeframe_seconds * data.info['sfreq']

            overlap_step = int(timepoints * overlap_percentage)

            n_samples = 0
            for i in range(0,int(data.get_data().shape[1]), int(timepoints - timepoints * overlap_percentage)):
                arr = data.get_data(start=i, stop=int(i+timepoints))
                if arr.shape[1] < timepoints:
                    continue
                if arr.shape != (19, timepoints):
                    continue

                patient_id = file.split(" ")[0]+file.split(" ")[1].replace("S", "")

                np.save(f"{save_folder}/{patient_id}_{i}.npy", arr)
                n_samples += 1
                if n_samples >= max_samples_per_patient:
                    break

        if split:

            files = glob.glob(f'{save_folder}/*')
            healthy = []
            depressed = []

            for file in files:
                id = file.split('/')[-1].split('_')[0]
                if id[0] == "H":
                    healthy.append(file)
                else:
                    depressed.append(file)

            depressed_ids = list(set([file.split('/')[-1].split('_')[0] for file in depressed]))
            healthy_ids = list(set([file.split('/')[-1].split('_')[0] for file in healthy]))

            # sort
            depressed_ids.sort()
            healthy_ids.sort()

            # shuffle with random seed
            random.seed(42)

            random.shuffle(depressed_ids)
            random.shuffle(healthy_ids)


            depressed_ids_train = depressed_ids[:int(len(depressed_ids)*0.8)]
            depressed_ids_val = depressed_ids[int(len(depressed_ids)*0.8):int(len(depressed_ids)*0.9)]
            depressed_ids_test = depressed_ids[int(len(depressed_ids)*0.9):]

            healthy_ids_train = healthy_ids[:int(len(healthy_ids)*0.8)]
            healthy_ids_val = healthy_ids[int(len(healthy_ids)*0.8):int(len(healthy_ids)*0.9)]
            healthy_ids_test = healthy_ids[int(len(healthy_ids)*0.9):]



            os.makedirs(f"{save_folder}/train")
            os.makedirs(f"{save_folder}/val")
            os.makedirs(f"{save_folder}/test")



            for file in depressed:
                id = file.split('/')[-1].split('_')[0]
                if id in depressed_ids_train:
                    # move file
                    shutil.move(file, f"{save_folder}/train")

                elif id in depressed_ids_val:
                    shutil.move(file, f"{save_folder}/val")
                elif id in depressed_ids_test:
                    shutil.move(file, f"{save_folder}/test")

            for file in healthy:
                id = file.split('/')[-1].split('_')[0]
                if id in healthy_ids_train:
                    shutil.move(file, f"{save_folder}/train")
                elif id in healthy_ids_val:
                    shutil.move(file, f"{save_folder}/val")
                elif id in healthy_ids_test:
                    shutil.move(file, f"{save_folder}/test")
