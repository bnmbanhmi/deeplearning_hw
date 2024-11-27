import numpy as np
import tensorflow as tf
import os
import cv2 as cv
from sklearn.model_selection import train_test_split
from tensorflow.python.training.tracking.tracking import Asset
from dataset import load_data, read_image, show_example, read_mask, convert2TfDataset
from config import *
from models.Unet import build_unet, compile_unet
from models.Unetmobilev2 import build_unet_mb2, compile_unet_mb2
# from models.r2attentionunet import build_unet_v2, build_r2_unet, build_att_unet, build_att_r2_unet, compile_unet_v2, compile_r2_unet, compile_att_r2_unet, compile_att_unet
from models.Attentionunet import build_att_unet, compile_att_unet
from models.AttentionUnetmobilev2 import build_att_unet_mb2, compile_att_unet_mb2
from models.Unetefficientb0 import build_unet_eff0, compile_unet_eff0
from models.AttentionUnetefficientb0 import build_att_unet_eff0, compile_att_unet_eff0
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback

def train():
    images, masks = load_data(IMAGE_PATH, MASK_PATH)
    print(f'Amount of images: {len(images)}')

    # Separate image paths into training_set, validation_set
    train_x, valid_x, train_y, valid_y = train_test_split(images, masks, test_size=0.2, random_state=42)
    print(f'Training: {len(train_x)} - Validation: {len(valid_x)}')

    train_step = len(train_x) // BATCH_SIZE
    if len(train_x) % BATCH_SIZE != 0:
        train_step += 1

    valid_step = len(valid_x) // BATCH_SIZE
    if len(valid_x) % BATCH_SIZE != 0:
        valid_step += 1

    print(f'{train_step} - {valid_step}')

    train_dataset = convert2TfDataset(train_x, train_y, BATCH_SIZE)
    valid_dataset = convert2TfDataset(valid_x, valid_y, BATCH_SIZE)

    # Initialize wandb
    wandb.init(
        project="my-awesome-project",  # Replace with your desired project name
        config={
            "model": MODEL_SELECTION,
            "batch_size": BATCH_SIZE,
            "epochs": 20,  # Or the number of epochs you are using
            # Add any other hyperparameters you want to track
        }
    )

    if MODEL_SELECTION == "unet":
        print("Using unet model")
        model = build_unet()
        model.summary()
        model, callbacks = compile_unet(model)
    elif MODEL_SELECTION == "unet_eff0":
        print("Using unet_eff0 model")
        model = build_unet_eff0()
        model.summary()
        model, callbacks = compile_unet_eff0(model)
    elif MODEL_SELECTION == "mb2_unet":
        print("Using mb2_unet model")
        model = build_unet_mb2()
        model.summary()
        model, callbacks = compile_unet_mb2(model)
    elif MODEL_SELECTION == "att_unet":
        print("Using att_unet model")
        model = build_att_unet()
        model.summary()
        model, callbacks = compile_att_unet(model)
    elif MODEL_SELECTION == "att_unet_mb2":
        print("Using att_unet_mb2 model")
        model = build_att_unet_mb2()
        model.summary()
        model, callbacks = compile_att_unet_mb2(model)
    elif MODEL_SELECTION == "unet_att_eff0":
        print("Using unet_att_eff0 model")
        model = build_att_unet_eff0()
        model.summary()
        model, callbacks = compile_att_unet_eff0(model)
    else:
        raise AssertionError("Model not supported. Currently support unet, unet_eff0, mb2_unet, att_unet, att_unet_mb2, unet_att_eff0")

    # Add WandbCallback to the callbacks list
    callbacks.append(WandbCallback())

    H = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        steps_per_epoch=train_step,
        validation_steps=valid_step,
        epochs=20,
        callbacks=callbacks
    )

    # Save the trained model
    model.save(os.path.join(SAVE_WEIGHTS_PATH, 'trained_model.h5'))

    # Finish the wandb run (optional, but recommended)
    wandb.finish()

    # Plotting the results
    fig = plt.figure()
    numOfEpoch = 20
    plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
    plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
    plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='accuracy')
    plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label='validation accuracy')
    plt.title('Accuracy and Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss|Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()