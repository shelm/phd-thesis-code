import argparse
import hcai_datasets
from tensorflow import keras
from base.base_handler import Handler


class TrainHandler(Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Train
        train_cfg = self.config["TRAIN"]
        self.epochs = train_cfg.getint("epochs")
        self.batch_size = train_cfg.getint("batch_size")
        self.loss = train_cfg["loss"]
        optimizer = train_cfg["optimizer"]
        self.optimizer = keras.optimizers.get(optimizer)
        self.optimizer.lr = train_cfg.getfloat("lr")
        self.metrics = train_cfg["keras_metrics"].split(";")

    @staticmethod
    def augmentation(ds):
        # TODO: Insert data augemantion
        return ds

    def exec(self):
        # Preparation before training
        ckpt_dir, log_dir, cfg_dir = self.prepare_directories_train()
        callbacks = self.prepare_callbacks()

        # Gather information
        ds_info = self.ds_info()
        n_output = (
            len(self.ds_label_filter)
            if self.ds_label_filter
            else len(ds_info.features[self.ds_label_id])
        )

        # Loading model with arguments
        output_shape = (n_output,)
        model_kwargs = {
            "output_shape": output_shape,
            "include_top": self.include_top,
            "pooling": self.pooling,
            "weights": self.weights,
        }
        model = self.load_model(self.model_name, model_kwargs)

        # Loading data
        ds_train, ds_info = self.load_ds_split(self.ds_train_split)
        ds_val, _ = self.load_ds_split(self.ds_val_split)

        # Applying preprocessing pipeline
        ds_train = self.transform_train(
            ds_train, model.preprocess_input, lambda x: x, self.batch_size
        )

        # Getting classweights
        class_weight = self.get_classweights(
            ds_info,
            self.ds_train_split.split("[")[0],
            self.ds_label_id,
            self.ds_label_filter,
        )

        # Training
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        history = model.fit(
            ds_train,
            epochs=self.epochs,
            validation_data=ds_val,
            class_weight=class_weight,
            callbacks=callbacks,
        )

        # Saving experiment data
        self.save_experiment_data(model, history)


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument(
        "-dc", "--dir_config", help="Path to the directory config file"
    )
    my_parser.add_argument(
        "-tc", "--train_config", help="Path to the training config file"
    )
    my_parser.add_argument(
        "-m",
        "--mode",
        default="train",
        type=str,
        choices=["train", "eval", "predict"],
        help="Execution mode: train, eval, predict",
    )
    args = my_parser.parse_args()

    handler = TrainHandler(args.dir_config, args.train_config)
    handler.exec()
