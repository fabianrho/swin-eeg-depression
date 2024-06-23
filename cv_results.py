import pandas as pd
import numpy as np 
import glob
import os
import time

def cv_results(savefolder):
    subfolders = [f.path for f in os.scandir(savefolder) if f.is_dir()]

    # sort subfolders
    subfolders.sort(key=lambda x: int(x.split('_')[-1]))

    folds_val_losses = []
    folds_val_accs = []

    for fold in subfolders:
        fold_results = pd.read_csv(fold + '/training_log.csv')
        # get row with lowest validation loss
        best_val_loss = fold_results.loc[fold_results['Validation Accuracy'].idxmax()]

        folds_val_losses.append(best_val_loss['Validation Loss'])
        folds_val_accs.append(best_val_loss['Validation Accuracy'])

    # print(np.mean(folds_val_losses), np.std(folds_val_losses))
    print("             ","Mean Accuracy:", round(np.mean(folds_val_accs),3), "std:", round(np.std(folds_val_accs), 3))

if __name__ == '__main__':
    
    for timewindow in [4,6]:
        for overlap in [0, 0.25, 0.5,0.75]:
            print("     time window:", timewindow, "overlap:", overlap)
            for model in ["swin", "deprnet"]:
                print("         ",model)
                savefolder = f'trained_models/{model}/{timewindow}s_{overlap}overlap'
                cv_results(savefolder)
                # time.sleep(0.5)
            print("\n")




    # savefolder = 'trained_models/swin/4s_0.25overlap'
    # cv_results(savefolder)
    
    