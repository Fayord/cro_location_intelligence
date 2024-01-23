import os
import torch


def main():
    import torch

    image_folder_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/data/full_image"
    image_name_list = os.listdir(image_folder_path)
    image_name_list.sort()
    image_name_list = [
        image_name for image_name in image_name_list if not image_name.startswith(".")
    ]
    image_path_list = [
        os.path.join(image_folder_path, image_name) for image_name in image_name_list
    ]
    import pandas as pd

    data_path = "/Users/user/Documents/Coding/cro_location_intelligence/notebook/7 BE for Ford.xlsx"
    df_dict = pd.read_excel(data_path, sheet_name=None)
    store_id_list = df_dict["Train"]["store_id"].tolist()
    store_id_list += df_dict["Test2022"]["store_id"].tolist()
    store_id_list = [str(i) for i in store_id_list]
    print(len(store_id_list))
    print(store_id_list[:5])
    store_id_we_have = []
    for image_name in image_name_list:
        store_id_we_have.append(image_name.split(".")[0])
    print(len(store_id_we_have))
    print(store_id_we_have[:5])
    store_id_we_dont_have = []
    for id in store_id_list:
        if id not in store_id_we_have:
            print(id)
            store_id_we_dont_have.append(id)

    print(len(store_id_we_dont_have))
    new_df_path = "/Users/user/Documents/Coding/cro_location_intelligence/src/data/all_data_embedding.csv"

    df = pd.read_csv(new_df_path)

    df.head()

    # use embedding to predict mockup_sale with xgboost
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    import numpy as np

    # linear regression
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_percentage_error

    # get all column has embedding_ prefix
    # feature_columns = [col for col in df.columns if col.startswith("embedding_")]
    # feature_columns += ["subset"]
    # feature_columns += ["y_nor"]
    # assign to x
    # all_data = df[feature_columns]
    all_data = df.copy()
    # normalize every column
    from sklearn.preprocessing import StandardScaler

    all_data = all_data.drop(columns=["store_id"])
    all_data = all_data.drop(columns=["y_nor"])
    train_all_data = all_data[all_data.subset == "train"]
    all_data = all_data.drop(columns=["subset"])
    train_all_data = train_all_data.drop(columns=["subset"])
    scaler = StandardScaler()
    scaler.fit(train_all_data)
    all_data = scaler.transform(all_data)
    all_data = pd.DataFrame(all_data)
    # all_data.columns = feature_columns
    # all_data["store_id"] = df["store_id"]
    all_data["y_nor"] = df["y_nor"]
    all_data["subset"] = df["subset"]
    all_data.head()

    # create train test with columns subset
    x_train = all_data[all_data.subset == "train"]
    x_train = x_train.drop(columns=["subset"])
    y_train = x_train["y_nor"]
    x_test = all_data[all_data.subset == "test"]
    x_test = x_test.drop(columns=["subset"])
    y_test = x_test["y_nor"]
    x_train = x_train.drop(columns=["y_nor"])
    x_test = x_test.drop(columns=["y_nor"])
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    X_train = np.array(x_train)
    X_test = np.array(x_test)

    # create validate set
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42
    )

    from typing import Any
    from lightning.pytorch.utilities.types import STEP_OUTPUT
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import lightning as pl
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import StepLR
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    from lightning.pytorch.loggers import TensorBoardLogger
    from torch.nn import Linear, BatchNorm1d, Dropout

    class MAPELoss(nn.Module):
        def forward(self, y_pred, y_true):
            epsilon = 1e-7
            percentage_error = torch.abs((y_true - y_pred) / (y_true + epsilon))
            mape = torch.mean(percentage_error) * 100.0
            return mape

    # Define a simple dataset
    class PriceDataset(Dataset):
        def __init__(self, features, labels, transform_y=None, transform_x=None):
            self.features = torch.tensor(features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32)
            self.transform_y = transform_y
            self.transform_x = transform_x

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            x = self.features[idx]
            y = self.labels[idx]
            if self.transform_x:
                x = self.transform_x(x)
            if self.transform_y:
                y = self.transform_y(y)
            return x, y

    import torch.nn.functional as F

    class PricePredictor(pl.LightningModule):
        def __init__(
            self,
            input_size,
            layer_sizes=[],
            dropout_rate=0.7,
            last_layer_dropout_rate=0.1,
            momentum=0.1,
            use_batch_norm=False,
            lr=0.00001,
        ):
            super(PricePredictor, self).__init__()

            layer_sizes = [input_size] + layer_sizes
            self.use_batch_norm = use_batch_norm
            self.layers = nn.ModuleList()
            self.batch_norms = nn.ModuleList()  # Use BatchNorm1d instead of LayerNorm
            self.dropouts = nn.ModuleList()

            for i in range(1, len(layer_sizes)):
                self.layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

                self.batch_norms.append(
                    nn.BatchNorm1d(layer_sizes[i], momentum=momentum)
                )

                # Use last_layer_dropout_rate for the last layer, otherwise use dropout_rate
                if i == len(layer_sizes) - 1:
                    self.dropouts.append(nn.Dropout(p=last_layer_dropout_rate))
                else:
                    self.dropouts.append(nn.Dropout(p=dropout_rate))

            self.lr = lr
            self.val_loss = 0
            self.training_step_loss = []

        def forward(self, x):
            for i in range(len(self.layers)):
                if self.use_batch_norm:
                    # Use BatchNorm1d instead of LayerNorm
                    x = F.leaky_relu(self.batch_norms[i](self.layers[i](x)))
                else:
                    x = F.leaky_relu(self.layers[i](x))

                x = self.dropouts[i](x)

            return x

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_pred = self(x)
            loss = nn.MSELoss()(y_pred.flatten(), y)
            self.training_step_loss.append(loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_pred = self(x)
            loss = nn.MSELoss()(y_pred.flatten(), y)
            self.val_loss = loss
            self.log("val_loss", loss)

        def test_step(self, batch, batch_idx):
            x, y = batch
            y_pred = self(x)
            loss = nn.MSELoss()(y_pred.flatten(), y)
            self.log("test_loss", loss)

        def on_train_epoch_end(self):
            # print("\n\tself.training_step_loss", self.training_step_loss)
            # print("\n\tself.training_step_loss", self.training_step_loss[0].shape)
            # print("\n\tself.training_step_loss", self.training_step_loss[0])
            # print()
            # avg_loss = torch.stack([x["loss"] for x in self.training_step_loss]).mean()
            avg_loss = torch.stack(
                [x.detach().cpu() for x in self.training_step_loss]
            ).mean()
            self.log("train_loss", avg_loss.item(), on_epoch=True)

    from lightning.pytorch.callbacks import LearningRateFinder

    class FineTuneLearningRateFinder(LearningRateFinder):
        def __init__(self, milestones, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.milestones = milestones
            self._min_lr = 1e-6
            self._max_lr = 1e-1
            # self._num_training_steps = 1000
            self._mode = "linear"

        def on_fit_start(self, *args, **kwargs):
            return

        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
                self.lr_find(trainer, pl_module)

    from lightning.pytorch.callbacks.early_stopping import EarlyStopping

    np.random.seed(42)

    class AddNoise(object):
        def __init__(self, noise_level=0.01):
            self.noise_level = noise_level

        def __call__(self, x):
            # random apply 20% of the time
            if torch.rand(1) < 0.2:
                return x
            noise = torch.randn_like(x) * self.noise_level
            augmented_x = x + noise
            return augmented_x

    transforms_x = AddNoise(noise_level=0.1)
    transforms_y = AddNoise(noise_level=0.025)
    # batch_size = 32
    batch_size = 256

    num_workers = 0
    train_dataset = PriceDataset(
        X_train, y_train, transform_x=transforms_x, transform_y=transforms_y
    )
    test_dataset = PriceDataset(X_test, y_test)
    val_dataset = PriceDataset(X_val, y_val)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_loader = test_loader

    from lightning.pytorch.callbacks import ModelCheckpoint

    dir_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{dir_path}/models/",
        save_top_k=2,
        monitor="val_loss",
        filename="best_model-{epoch}",
    )

    # Initialize the model
    input_size = X_train.shape[1:]
    model = PricePredictor(input_size[0], layer_sizes=[512, 128, 32, 1]).to(device)
    # check_point_path = "/Users/user/Documents/Coding/cro_location_intelligence/src/models/pretrain.ckpt"
    # model.load_from_checkpoint(check_point_path)
    # model = PricePredictor.load_from_checkpoint(
    #     check_point_path,
    #     input_size=input_size[0],
    #     layer_sizes=[512, 128, 32, 1],
    #     use_batch_norm=False,
    #     dropout_rate=0.2,
    # ).to(device)

    # model.eval()
    # Initialize a PyTorch Lightning Trainer
    logger = TensorBoardLogger(f"{dir_path}/tb_logs", name="my_model")

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=50, verbose=False, mode="min"
    )
    trainer = pl.Trainer(
        callbacks=[
            # FineTuneLearningRateFinder(
            #     milestones=(
            #         100,
            #         1000,
            #         2000,
            #         3000,
            #         5000,
            #         10000,
            #     )
            # ),
            early_stop_callback,
            checkpoint_callback,
        ],
        min_epochs=20000,
        max_epochs=30000,
        logger=logger,
        precision="16-mixed",
        accelerator="mps",
    )
    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test the model
    trainer.test(dataloaders=test_loader)

    # Make predictionss
    model.to(device)
    model.eval()
    with torch.no_grad():
        example_input = torch.tensor(X_test[:5], dtype=torch.float32).to(device)
        predictions = model(example_input).flatten().cpu().numpy()

    print("Example Predictions:", predictions)
    for i, (path, _) in enumerate(trainer.checkpoint_callback.best_k_models.items()):
        model = PricePredictor.load_from_checkpoint(path, input_size=input_size[0])
        PATH = f"./models/{i}th_best.pt"
        torch.save(model, PATH)
        print(PATH)
        # break


if __name__ == "__main__":
    main()
