from data_loader import EEGDataset
import torch
from transformers import AutoImageProcessor, Swinv2ForImageClassification, ResNetForImageClassification, DeiTForImageClassification
from transformers import ViTConfig, ViTForImageClassification
import transformers
from tqdm import tqdm
import numpy as np
import os
import shutil


from models.swin_v2 import SwinTransformerV2, load_pretrained
from models.deprnet import DeprNet

import matplotlib.pyplot as plt


class EEG_Depression_Dectection():

    def __init__(self, data_folder, save_folder, model_type, evaluation = False, cross_validation = True, pretrained = False, resize_to = 256):

        self.data_folder = data_folder
        self.save_folder = save_folder
        self.model_type = model_type
        self.pretrained = pretrained
        self.resize_to = resize_to

        self.cross_validation = cross_validation

        if not evaluation:

            if os.path.exists(self.save_folder) and not self.cross_validation:
                raise ValueError("Save folder already exists")
            
            elif os.path.exists(self.save_folder) and self.cross_validation:
                pass
            else:
                os.makedirs(self.save_folder)

                if not self.cross_validation:
                    # create subfolders for checkpoints and validation visualisation
                    self.checkpoint_folder = os.path.join(self.savefolder, "checkpoints")
                    self.validation_visualisation_folder = os.path.join(self.savefolder, "validation_visualisation")
                    if not os.path.exists(self.checkpoint_folder):
                        os.mkdir(self.checkpoint_folder)


        self.evaluation = evaluation
            



    def train_cv(self, folds_path = "data/folds.txt", epochs = 100, early_stopping_threshold = None, device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")):

        self.device = device
        # load fold ids
        with open(folds_path) as f:
            lines = f.readlines()
            # remove \n
            lines = [line.replace("\n","") for line in lines]
            folds = [line.split(",") for line in lines]

        self.all_ids = [id for fold in folds for id in fold]

        for self.fold_n, fold in enumerate(folds):

            if os.path.exists(os.path.join(self.save_folder, f"fold_{self.fold_n}")):
                if os.path.exists(os.path.join(self.save_folder, f"fold_{self.fold_n}", "model.pth")):
                    print(f"Fold {self.fold_n} already trained")
                    continue
                else:
                    # delete fold folder if it did not finish training (restart training)
                    print(f"Restarting training for fold {self.fold_n}")
                    shutil.rmtree(os.path.join(self.save_folder, f"fold_{self.fold_n}"))
                    
            self.train_cv_fold(fold, epochs, early_stopping_threshold)


        



    def train_cv_fold(self, validation_ids, epochs = 100, early_stopping_threshold = None):

        self.fold_save_folder = os.path.join(self.save_folder, f"fold_{self.fold_n}")
        os.makedirs(self.fold_save_folder)

        # create subfolders for checkpoints and validation visualisation
        self.checkpoint_folder = os.path.join(self.fold_save_folder, "checkpoints")
        if not os.path.exists(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)


        self.get_model()

        self.model.to(self.device)

        self.epochs_since_improvement = 0
        self.previous_val_accuracy = 0
    

        self.load_cv_data(validation_ids)


        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = torch.nn.BCELoss()
        # if self.model_type == "swin":
        #     # self.criterion = torch.nn.BCEWithLogitsLoss()
        #     self.criterion = torch.nn.BCELoss()
        #     # label smoothing

        #     # self.criterion = torch.nn.CrossEntropyLoss()
        # else:
        #     self.criterion = torch.nn.CrossEntropyLoss()
        #     # self.criterion = torch.nn.BCELoss()
        #     # self.criterion = torch.nn.BCEWithLogitsLoss()


        for self.epoch in range(1, epochs+1):
            self.train_losses = []
            self.train_accuracies = []

            with tqdm(self.dataloader_train, unit="batch") as tepoch:
                for data in tepoch:
                    image,label = data

                    image = image.to(self.device)
                    label = label.to(self.device)

                    if self.model_type == "1dtransformer":
                        output = self.model(image).squeeze(1)
                    elif self.model_type in ["resnet", "deit", "vit"]:
                        output = self.model(image).logits.squeeze(1)

                    else:
                        # output = torch.softmax(self.model(image), dim = 1)#.squeeze(1)
                        output = self.model(image)

                        if self.model_type == "swin":
                            output = torch.nn.Softmax(dim=1)(output)


                    binary_output = (output > 0.5).float()
                    accuracy = (binary_output == label).float().cpu().numpy()
                    self.train_accuracies.extend(accuracy)

                    loss = self.criterion(output, label)
                    # loss.requires_grad = True
                    loss.backward()
                    self.train_losses.append(loss.item())
                    optimizer.step()
                    optimizer.zero_grad()

                    # print(output, label, loss.item())
                    tepoch.set_postfix(avg_loss = sum(self.train_losses)/len(self.train_losses), loss = loss.item(), accuracy = np.mean(self.train_accuracies))


            self.on_epoch_end()

            if np.mean(self.val_accuracy) > self.previous_val_accuracy:
                self.epochs_since_improvement = 0
                self.previous_val_accuracy = np.mean(self.val_accuracy)
            else:
                self.epochs_since_improvement += 1

            if early_stopping_threshold is not None:
                if self.epochs_since_improvement == early_stopping_threshold:
                    print("Early stopping\n")
                    break

        # save model
        torch.save(self.model.state_dict(), os.path.join(self.fold_save_folder, "model.pth"))



    def get_model(self):

        match self.model_type:
            case "swin":
                if self.pretrained:
                    self.model = SwinTransformerV2(img_size = 256, num_classes=2, in_chans = 3, window_size=8)
                    self.model = load_pretrained(self.model, "/home/u887755/bci/swinv2_tiny_patch4_window8_256.pth")

                    # self.model = SwinTransformerV2(img_size = 256, num_classes=1, in_chans = 3, window_size=16)
                    # self.model = load_pretrained(self.model, "/home/u887755/bci/swinv2_tiny_patch4_window16_256.pth")
                else:

                    # swin-t
                    self.model = SwinTransformerV2(img_size = 256, num_classes=2, in_chans = 3, window_size=8)

                    # swin-s
                    # self.model = SwinTransformerV2(img_size = 256, num_classes=1, in_chans = 1, window_size=8,
                                                #    depths=[2,2,18,2], num_heads=[3,6,12,24], embed_dim=96)


                    # swin-b
                    # self.model = SwinTransformerV2(img_size = 256, num_classes=1, in_chans = 1, window_size=8,
                    #                                depths = [2,2,18,2], num_heads=[4,8,16,32], embed_dim=128)

            case "deprnet":

                self.model = DeprNet()


            case "resnet":
                config = transformers.ResNetConfig.from_pretrained("microsoft/resnet-50")

                config.num_channels = 1
                config.num_labels = 1

                self.model = ResNetForImageClassification(config)


    

    def load_cv_data(self, validation_ids: list):

        # if self.pretrained and self.model_type == "swin":
        if self.model_type == "swin":

            three_channels = True
        else:
            three_channels = False

        train_ids = [id for id in self.all_ids if id not in validation_ids]

        dataset_train = EEGDataset(f"{self.data_folder}", file_ids = train_ids, resize_to=(self.resize_to,self.resize_to), three_channels = three_channels)
        self.dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True, drop_last=True)

        if self.evaluation:
            val_batch_size = 64
        else:
            val_batch_size = 32
        dataset_val = EEGDataset(f"{self.data_folder}", file_ids = validation_ids, resize_to=(self.resize_to,self.resize_to), three_channels = three_channels)
        self.dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=val_batch_size, shuffle=False, drop_last=False)


    def load_data(self):

        if self.pretrained and self.model_type == "swin":
            three_channels = True

        dataset_train = EEGDataset(f"{self.data_folder}/train", resize_to=(self.resize_to,self.resize_to), three_channels = three_channels)
        # subset
        dataset_train = torch.utils.data.Subset(dataset_train, range(1000))
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=12, shuffle=True, drop_last=True)

        dataset_val = EEGDataset(f"{self.data_folder}/val", resize_to=(self.resize_to,self.resize_to), three_channels = three_channels)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=24, shuffle=False, drop_last=False)




    def on_epoch_end(self):
        self.validation()
        self.save_checkpoint()
        self.save_history()
        if self.epoch > 1:
            self.plot_history()

        

    def validation(self):
        """ Callback function to calculate the validation loss and dice score after each epoch
        """
        self.model.eval()
        # self.criterion = torch.nn.CrossEntropyLoss()


        # if self.model_type == "swin":
        #     # self.criterion = torch.nn.BCEWithLogitsLoss()
        #     self.criterion = torch.nn.BCELoss()
        # else:
        #     # self.criterion = torch.nn.CrossEntropyLoss()
        #     self.criterion = torch.nn.BCEWithLogitsLoss()

        self.val_losses = []
        self.val_accuracy = []

        for data in self.dataloader_val:
            image,label = data

            image = image.to(self.device)
            label = label.to(self.device)

            if self.model_type == "1dtransformer":
                output = self.model(image).squeeze(1)
            elif self.model_type in ["resnet", "deit", "vit"]:
                output = self.model(image).logits.squeeze(1)
            else:
                # output = torch.sigmoid(self.model(image)).squeeze(1)
                # output = torch.softmax(self.model(image), dim = 1)#.squeeze(1)
                output = self.model(image)

                if self.model_type == "swin":
                    output = torch.nn.Softmax(dim=1)(output)

                # print(output)
            self.val_out = output
            self.val_label = label



            binary_output = (output > 0.5).float()
            accuracy = (binary_output == label).float().cpu().numpy()
            self.val_accuracy.extend(accuracy)

            loss = self.criterion(output, label)

            self.val_losses.append(loss.item())
        # print(self.val_losses)

        # print(self.val_losses)
        # print("Validation loss: ", sum(self.val_losses)/len(self.val_losses), "Validation accuracy: ", np.mean(self.val_accuracy))
        print("Validation loss: ", np.mean(self.val_losses), "Validation accuracy: ", np.mean(self.val_accuracy))



    def save_checkpoint(self):
        if self.epoch == 1:  
            self.best_accuracy = np.mean(self.val_accuracy)
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_folder, f"epoch_{self.epoch}_{self.best_accuracy:.4f}.pth"))

        else:
            if torch.mean(torch.tensor(np.array(self.val_accuracy))) > self.best_accuracy:
                self.best_accuracy = torch.mean(torch.tensor(np.array(self.val_accuracy)))
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_folder, f"epoch_{self.epoch}_{self.best_accuracy:.4f}.pth"))

    def save_history(self):


        if self.cross_validation:
            save_folder = self.fold_save_folder
        else:
            save_folder = self.savefolder



        if os.path.exists(os.path.join(save_folder, "training_log.csv")):
            with open(os.path.join(save_folder, "training_log.csv"), "a") as f:
                f.write(f"{self.epoch},{torch.mean(torch.tensor(np.array(self.train_losses)))},{torch.mean(torch.tensor(np.array(self.train_accuracies)))},{torch.mean(torch.tensor(np.array(self.val_losses)))},{torch.mean(torch.tensor(np.array(self.val_accuracy)))}\n")
                # f.write(f"{self.epoch},{torch.mean(self.train_losses)},{torch.mean(self.train_accuracies)},{torch.mean(self.val_losses)},{torch.mean(self.val_accuracy)}\n")

        else:
            with open(os.path.join(save_folder, "training_log.csv"), "w") as f:
                f.write("Epoch,Train Loss,Train Accuracy,Validation Loss,Validation Accuracy\n")
                # f.write(f"{self.epoch},{torch.mean(torch.tensor(self.train_losses))},{torch.mean(torch.tensor(self.train_accuracies))},{torch.mean(torch.tensor(self.val_losses))},{torch.mean(torch.tensor(self.val_accuracy))}\n")
                f.write(f"{self.epoch},{torch.mean(torch.tensor(np.array(self.train_losses)))},{torch.mean(torch.tensor(np.array(self.train_accuracies)))},{torch.mean(torch.tensor(np.array(self.val_losses)))},{torch.mean(torch.tensor(np.array(self.val_accuracy)))}\n")

                # f.write(f"{self.epoch},{torch.mean(self.train_losses)},{torch.mean(self.train_accuracies)},{torch.mean(self.val_losses)},{torch.mean(self.val_accuracy)}\n")

    def plot_history(self):
        """ Callback function to plot the training history
        """

        if self.cross_validation:
            save_folder = self.fold_save_folder
        else:
            save_folder = self.savefolder

        history = np.genfromtxt(os.path.join(save_folder, "training_log.csv"), delimiter=",", skip_header=1)
        fig, ax = plt.subplots(2, 1, figsize=(15, 15))
        # increase font size
        plt.rcParams.update({'font.size': 18})
        # make x axis integer
        ax[0].plot(history[:, 0], history[:, 1], label="Train Loss")
        ax[0].plot(history[:, 0], history[:, 3], label="Validation Loss")
        ax[0].set_title("Loss")
        ax[0].legend()
        ax[0].xaxis.get_major_locator().set_params(integer=True)

        ax[1].plot(history[:, 0], history[:, 2], label="Train Accuracy")
        ax[1].plot(history[:, 0], history[:, 4], label="Validation Accuracy")
        ax[1].set_title("Accuracy")
        ax[1].legend()
        ax[1].xaxis.get_major_locator().set_params(integer=True)

        plt.savefig(os.path.join(save_folder, "history.png"))
        plt.close(fig)



    def evaluation_cv(self, metric, folds_path = "data/folds.txt", device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")):

        self.device = device
        # load fold ids
        with open(folds_path) as f:
            lines = f.readlines()
            # remove \n
            lines = [line.replace("\n","") for line in lines]
            folds = [line.split(",") for line in lines]

        self.all_ids = [id for fold in folds for id in fold]

        metric_scores = []

        for self.fold_n, fold in enumerate(folds):
                
            fold_metric = self.evaluation_cv_fold(fold, metric)
            metric_scores.append(fold_metric)

        metrics_mean = np.mean(metric_scores, axis = 0)
        metrics_std = np.std(metric_scores, axis = 0)

        return metrics_mean, metrics_std
        
        # print(f"Mean {metric}: ", np.mean(metric_scores, axis = 0), f"Std {metric}: ", np.std(metric_scores, axis = 0))


    def evaluation_cv_fold(self, fold, metric):
        self.fold_save_folder = os.path.join(self.save_folder, f"fold_{self.fold_n}")

        self.pretrained = False
        self.get_model()

        # get path of best checkpoint
        checkpoints = os.listdir(os.path.join(self.fold_save_folder, "checkpoints"))
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))
        best_checkpoint = checkpoints[-1]

        # load model
        self.model.load_state_dict(torch.load(os.path.join(self.fold_save_folder, "checkpoints", best_checkpoint)))

        self.model.to(self.device)

        self.load_cv_data(fold)

        self.model.eval()


        val_metric = []
        val_predictions = []
        val_labels = []

        for data in tqdm(self.dataloader_val):
            image,label = data

            image = image.to(self.device)
            label = label.to(self.device)

            if self.model_type == "1dtransformer":
                output = self.model(image).squeeze(1)
            elif self.model_type in ["resnet", "deit", "vit"]:
                output = self.model(image).logits.squeeze(1)
            else:
                output = self.model(image)

                if self.model_type == "swin":
                    output = torch.nn.Softmax(dim=1)(output)

                # print(output)
            self.val_out = output
            self.val_label = label



            binary_output = (output > 0.5).float()

            binary_output = torch.argmax(binary_output, dim=1)
            label = torch.argmax(label, dim=1)


            val_predictions.extend(binary_output.cpu().numpy())
            val_labels.extend(label.cpu().numpy())
        val_labels = torch.tensor(val_labels)
        val_predictions = torch.tensor(val_predictions)

        

        match metric:
            case "accuracy":
                # one hot to binary
                # binary_output = torch.argmax(binary_output, dim=1)
                # label = torch.argmax(label, dim=1)

                val_metric = np.mean((val_predictions == val_labels).float().cpu().numpy())
            case "precision":
                # binary_output = torch.argmax(binary_output, dim=1)
                # label = torch.argmax(label, dim=1)
                # TP = torch.sum(label * binary_output)
                # FP = torch.sum(binary_output) - TP

                TP = torch.sum(val_labels * val_predictions)
                FP = torch.sum(val_predictions) - TP
                val_metric = TP / (TP + FP)
                # # print if precision is nan
                # if not torch.isnan(precision):
                #     val_metric.append(precision.item())

            case "recall":
                TP = torch.sum(val_labels * val_predictions)
                FP = torch.sum(val_predictions) - TP
                val_metric = TP / (TP + FN)
                # if not torch.isnan(recall):
                #     val_metric.append(recall.item())
            case "f1":
                TP = torch.sum(val_labels * val_predictions)
                FP = torch.sum(val_predictions) - TP
                FN = torch.sum(val_labels) - TP
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                val_metric = 2 * (precision * recall) / (precision + recall)
                # if not torch.isnan(f1):
                #     val_metric.append(f1.item())

            case "all":
                TP = torch.sum(val_labels * val_predictions)
                FP = torch.sum(val_predictions) - TP
                FN = torch.sum(val_labels) - TP
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * (precision * recall) / (precision + recall)
                accuracy = np.mean((val_predictions == val_labels).float().cpu().numpy())
                val_metric = [accuracy, precision.item(), recall.item(), f1.item()]

        print(val_metric)
        return val_metric


        # print("Validation accuracy: ", np.mean(self.val_accuracy))





        


if __name__ == "__main__":
    # training = EEG_Depression_Dectection(data_folder="data/data_6s_0overlap", save_folder="trained_models/small_6s_0overlapCV", model_type="swin", cross_validation=True, resize_to=256, pretrained=True)
    training = EEG_Depression_Dectection(data_folder="data/data_7s_0overlap", save_folder="trained_models/TEMP", model_type="swin", cross_validation=True, resize_to=256, pretrained=True)

    training.train_cv("data/folds.txt", epochs = 100, early_stopping_threshold = 5, device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))