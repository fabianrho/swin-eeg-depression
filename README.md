# EEG-based Major Depression Disorder Recognition using Swin Transformers

Instructions to run the experiments:
- Download the pretrained Swinv2-Tiny weights from the [official Swin Transformer repository](https://github.com/microsoft/Swin-Transformer#:~:text=1K%20model-,SwinV2%2DT,-ImageNet%2D1K).
- Download the [MDD dataset](https://figshare.com/articles/dataset/EEG_Data_New/4244171) and extract to folder `data_mdd`
- Run `data_preprocessing.py` to extract data samples
- Use `.sh` files in folder `train` to train models.
