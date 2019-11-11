import time

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .dragon_utils import (
    binary_classification_loss,
    dead_loss,
    dragonnet_loss_binarycross,
    make_dragonnet,
    make_ned,
    make_tarnet,
    make_tarreg_loss,
    ned_loss,
    post_cut,
    regression_loss,
    track_epsilon,
    treatment_accuracy,
)


def _split_output(yt_hat, t, y, y_scaler, x, index):
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].copy())
    g = yt_hat[:, 2].copy()

    if yt_hat.shape[1] == 4:
        eps = yt_hat[:, 3][0]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    y = y_scaler.inverse_transform(y.copy())
    var = "average propensity for treated: {} and untreated: {}".format(
        g[t.squeeze() == 1.0].mean(), g[t.squeeze() == 0.0].mean()
    )
    print(var)

    return {
        "q_t0": q_t0,
        "q_t1": q_t1,
        "g": g,
        "t": t,
        "y": y,
        "x": x,
        "index": index,
        "eps": eps,
    }


def train_dragon(
    t,
    y_unscaled,
    x,
    targeted_regularization=True,
    knob_loss=dragonnet_loss_binarycross,
    ratio=1.0,
    val_split=0.1,
    batch_size=512,
    num_epochs=100,
    learning_rate=0.001,
    verbose=1,
):
    """Build and Train the DragonNet Model on given data

    :param t: treatment treatment to train on
    :param y_unscaled: outcomes to train on
    :param x: covariates to train on
    :param targeted_regularization: use targeted regularization?
    :param knob_loss: base loss to use for tar_reg
    :param ratio:
    :param val_split:
    :param batch_size: batch size to use
    :param num_epochs: number of epochs to run
    :return:
    """

    # Build Model structure
    dragonnet = make_dragonnet(x.shape[1], 0.01)

    # Only report regression loss for now
    metrics = [regression_loss]

    if targeted_regularization:
        loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
    else:
        loss = knob_loss

    yt = np.c_[y_unscaled, t]

    start_time = time.time()

    dragonnet.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)

    class EarlyStoppingByLossVal(Callback):
        def __init__(self, monitor="regression_loss", value=400, verbose=0):
            super(Callback, self).__init__()
            self.monitor = monitor
            self.value = value
            self.verbose = verbose

        def on_epoch_end(self, epoch, logs={}):
            current = logs.get(self.monitor)
            if current < self.value:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True

    adam_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="regression_loss", patience=10, min_delta=0.0),
        ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=5,
            verbose=verbose,
            mode="auto",
            min_delta=1e-8,
            cooldown=0,
            min_lr=0,
        )
    ]

    dragonnet.fit(
        x,
        yt,
        callbacks=adam_callbacks,
        validation_split=val_split,
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    elapsed_time = time.time() - start_time
    if verbose:
        print("***************************** elapsed_time is: ", elapsed_time)

    return dragonnet


def train_and_predict_dragons(
    t,
    y_unscaled,
    x,
    targeted_regularization=True,
    output_dir="",
    knob_loss=dragonnet_loss_binarycross,
    ratio=1.0,
    dragon=1,
    val_split=0.1,
    batch_size=512,
):

    verbose = 1
    y_scaler = StandardScaler().fit(y_unscaled)
    y = y_scaler.transform(y_unscaled)
    train_outputs = []
    test_outputs = []
    runs = 1
    for i in range(runs):
        if dragon == 0:

            dragonnet = make_tarnet(x.shape[1], 0.01)
        elif dragon == 1:
            dragonnet = make_dragonnet(x.shape[1], 0.01)

        metrics = [
            regression_loss,
            binary_classification_loss,
            treatment_accuracy,
            track_epsilon,
        ]

        if targeted_regularization:
            loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
        else:
            loss = knob_loss

        tf.random.set_random_seed(i)
        np.random.seed(i)
        train_index, test_index = train_test_split(
            np.arange(x.shape[0]), test_size=0, random_state=1
        )
        test_index = train_index

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        t_train, t_test = t[train_index], t[test_index]
        yt_train = np.concatenate([y_train, t_train], 1)

        start_time = time.time()

        dragonnet.compile(optimizer=Adam(lr=1e-3), loss=loss, metrics=metrics)

        adam_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor="val_loss", patience=2, min_delta=0.0),
            ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=5,
                verbose=verbose,
                mode="auto",
                min_delta=1e-8,
                cooldown=0,
                min_lr=0,
            ),
        ]

        dragonnet.fit(
            x_train,
            yt_train,
            callbacks=adam_callbacks,
            validation_split=val_split,
            epochs=100,
            batch_size=batch_size,
            verbose=verbose,
        )

        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor="val_loss", patience=40, min_delta=0.0),
            ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=5,
                verbose=verbose,
                mode="auto",
                min_delta=0.0,
                cooldown=0,
                min_lr=0,
            ),
        ]

        # should pick something better!
        sgd_lr = 1e-5
        momentum = 0.9
        dragonnet.compile(
            optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True),
            loss=loss,
            metrics=metrics,
        )
        dragonnet.fit(
            x_train,
            yt_train,
            callbacks=sgd_callbacks,
            validation_split=val_split,
            epochs=300,
            batch_size=batch_size,
            verbose=verbose,
        )

        elapsed_time = time.time() - start_time
        print("***************************** elapsed_time is: ", elapsed_time)

        yt_hat_test = dragonnet.predict(x_test)
        yt_hat_train = dragonnet.predict(x_train)

        test_outputs += [
            _split_output(yt_hat_test, t_test, y_test, y_scaler, x_test, test_index)
        ]
        train_outputs += [
            _split_output(
                yt_hat_train, t_train, y_train, y_scaler, x_train, train_index
            )
        ]
        K.clear_session()

    return test_outputs, train_outputs


def train_and_predict_ned(
    t,
    y_unscaled,
    x,
    targeted_regularization=True,
    output_dir="",
    knob_loss=dragonnet_loss_binarycross,
    ratio=1.0,
    dragon=1,
    val_split=0.1,
    batch_size=512,
):

    verbose = 0
    y_scaler = StandardScaler().fit(y_unscaled)
    y = y_scaler.transform(y_unscaled)

    train_outputs = []
    test_outputs = []
    runs = 25
    for i in range(runs):

        nednet = make_ned(x.shape[1], 0.01)

        metrics_ned = [ned_loss]
        metrics_cut = [regression_loss]

        tf.random.set_random_seed(i)
        np.random.seed(i)
        train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=0.0)
        test_index = train_index
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        t_train, t_test = t[train_index], t[test_index]
        yt_train = np.concatenate([y_train, t_train], 1)

        start_time = time.time()

        nednet.compile(optimizer=Adam(lr=1e-3), loss=ned_loss, metrics=metrics_ned)

        adam_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor="val_loss", patience=2, min_delta=0.0),
            ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=5,
                verbose=verbose,
                mode="auto",
                min_delta=1e-8,
                cooldown=0,
                min_lr=0,
            ),
        ]

        nednet.fit(
            x_train,
            yt_train,
            callbacks=adam_callbacks,
            validation_split=val_split,
            epochs=100,
            batch_size=batch_size,
            verbose=verbose,
        )

        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor="val_loss", patience=40, min_delta=0.0),
            ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=5,
                verbose=verbose,
                mode="auto",
                min_delta=0.0,
                cooldown=0,
                min_lr=0,
            ),
        ]

        sgd_lr = 1e-5
        momentum = 0.9
        nednet.compile(
            optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True),
            loss=ned_loss,
            metrics=metrics_ned,
        )
        print(nednet.summary())
        nednet.fit(
            x_train,
            yt_train,
            callbacks=sgd_callbacks,
            validation_split=val_split,
            epochs=300,
            batch_size=batch_size,
            verbose=verbose,
        )

        t_hat_test = nednet.predict(x_test)[:, 1]
        t_hat_train = nednet.predict(x_train)[:, 1]

        # cutting the activation layer
        cut_net = post_cut(nednet, x.shape[1], 0.01)

        cut_net.compile(optimizer=Adam(lr=1e-3), loss=dead_loss, metrics=metrics_cut)

        adam_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor="val_loss", patience=2, min_delta=0.0),
            ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=5,
                verbose=verbose,
                mode="auto",
                min_delta=1e-8,
                cooldown=0,
                min_lr=0,
            ),
        ]
        print(cut_net.summary())

        cut_net.fit(
            x_train,
            yt_train,
            callbacks=adam_callbacks,
            validation_split=val_split,
            epochs=100,
            batch_size=batch_size,
            verbose=verbose,
        )

        elapsed_time = time.time() - start_time
        print("***************************** elapsed_time is: ", elapsed_time)

        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor="val_loss", patience=40, min_delta=0.0),
            ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=5,
                verbose=verbose,
                mode="auto",
                min_delta=0.0,
                cooldown=0,
                min_lr=0,
            ),
        ]

        sgd_lr = 1e-5
        momentum = 0.9
        cut_net.compile(
            optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True),
            loss=dead_loss,
            metrics=metrics_cut,
        )

        cut_net.fit(
            x_train,
            yt_train,
            callbacks=sgd_callbacks,
            validation_split=val_split,
            epochs=300,
            batch_size=batch_size,
            verbose=verbose,
        )

        y_hat_test = cut_net.predict(x_test)
        y_hat_train = cut_net.predict(x_train)

        yt_hat_test = np.concatenate([y_hat_test, t_hat_test.reshape(-1, 1)], 1)
        yt_hat_train = np.concatenate([y_hat_train, t_hat_train.reshape(-1, 1)], 1)

        test_outputs += [
            _split_output(yt_hat_test, t_test, y_test, y_scaler, x_test, test_index)
        ]
        train_outputs += [
            _split_output(
                yt_hat_train, t_train, y_train, y_scaler, x_train, train_index
            )
        ]
        K.clear_session()

    return test_outputs, train_outputs
