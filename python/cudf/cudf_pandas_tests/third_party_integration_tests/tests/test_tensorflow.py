# Copyright (c) 2023-2024, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

pytestmark = pytest.mark.assert_eq(fn=pd._testing.assert_equal)


@pytest.fixture(scope="module")
def df():
    rng = np.random.RandomState(42)

    nrows = 303
    columns = {
        "age": rng.randint(29, 78, size=(nrows,), dtype="int64"),
        "sex": rng.randint(0, 2, size=(nrows,), dtype="int64"),
        "cp": rng.randint(0, 5, size=(nrows,), dtype="int64"),
        "trestbps": rng.randint(94, 201, size=(nrows,), dtype="int64"),
        "chol": rng.randint(126, 565, size=(nrows,), dtype="int64"),
        "fbs": rng.randint(0, 2, size=(nrows,), dtype="int64"),
        "restecg": rng.randint(0, 3, size=(nrows,), dtype="int64"),
        "thalach": rng.randint(71, 203, size=(nrows,), dtype="int64"),
        "exang": rng.randint(0, 2, size=(nrows,), dtype="int64"),
        "oldpeak": rng.uniform(0.0, 6.2, size=(nrows,)),
        "slope": rng.randint(1, 4, size=(nrows,), dtype="int64"),
        "ca": rng.randint(0, 4, size=(nrows,), dtype="int64"),
        "thal": rng.choice(
            ["fixed", "normal", "reversible", "1", "2"], size=(nrows,)
        ),
        "target": rng.randint(0, 2, size=(nrows,), dtype="int64"),
    }

    return pd.DataFrame(columns)


@pytest.fixture(scope="module")
def target(df):
    return df.pop("target")


@pytest.fixture
def model_gen():
    def make_model(numeric_features):
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(numeric_features)
        model = tf.keras.Sequential(
            [
                normalizer,
                tf.keras.layers.Dense(10, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model

    return make_model


def test_dataframe_as_array(model_gen, df, target):
    tf.keras.utils.set_random_seed(42)

    numeric_feature_names = ["age", "thalach", "trestbps", "chol", "oldpeak"]
    numeric_features = df[numeric_feature_names]

    numeric_features = tf.convert_to_tensor(
        numeric_features.values, dtype=tf.float32
    )

    model = model_gen(numeric_features)
    model.fit(numeric_features, target, epochs=1, batch_size=BATCH_SIZE)

    test_data = numeric_features[:BATCH_SIZE]
    return model.predict(test_data)


def test_dataframe_as_dataset(model_gen, df, target):
    tf.keras.utils.set_random_seed(42)

    numeric_feature_names = ["age", "thalach", "trestbps", "chol", "oldpeak"]
    numeric_features = df[numeric_feature_names]

    numeric_features = tf.convert_to_tensor(
        numeric_features.values, dtype=tf.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))
    dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)

    model = model_gen(numeric_features)
    model.fit(dataset, epochs=1)

    test_data = dataset.take(1)
    return model.predict(test_data)


def stack_dict(inputs, func=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
        values.append(CastLayer()(inputs[key]))

    class MyLayer(tf.keras.layers.Layer):
        def call(self, val):
            return func(val, axis=-1)

    return MyLayer()(values)


def test_dataframe_as_dictionary_with_keras_input_layer(df, target):
    # ensure deterministic results
    tf.keras.utils.set_random_seed(42)

    numeric_feature_names = ["age", "thalach", "trestbps", "chol", "oldpeak"]
    numeric_features = df[numeric_feature_names]

    inputs = {}
    for name in numeric_features:
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=tf.float32)

    x = stack_dict(inputs, func=tf.concat)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(stack_dict(dict(numeric_features)))

    x = normalizer(x)
    x = tf.keras.layers.Dense(10, activation="relu")(x)
    x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, x)

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
        run_eagerly=True,
    )

    # Train with dictionary of columns as input:
    model.fit(dict(numeric_features), target, epochs=1, batch_size=BATCH_SIZE)

    # Train with a dataset of dictionary-elements
    numeric_dict_ds = tf.data.Dataset.from_tensor_slices(
        (dict(numeric_features), target)
    )
    numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(
        BATCH_SIZE
    )
    model.fit(numeric_dict_batches, epochs=1)

    # Predict
    return model.predict(numeric_dict_batches.take(1))


def test_full_example_train_with_ds(df, target):
    # https://www.tensorflow.org/tutorials/load_data/pandas_dataframe#full_example
    # Inputs are converted to tf.dataset and then batched

    # ensure deterministic results
    tf.keras.utils.set_random_seed(42)

    numeric_feature_names = ["age", "thalach", "trestbps", "chol", "oldpeak"]
    binary_feature_names = ["sex", "fbs", "exang"]
    categorical_feature_names = ["cp", "restecg", "slope", "thal", "ca"]

    numeric_features = df[numeric_feature_names]

    inputs = {}
    for name, column in df.items():
        if isinstance(column[0], str):
            dtype = tf.string
        elif name in categorical_feature_names or name in binary_feature_names:
            dtype = tf.int64
        else:
            dtype = tf.float32

        inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

    preprocessed = []

    # Process binary features
    for name in binary_feature_names:
        inp = inputs[name]
        inp = inp[:, tf.newaxis]
        float_value = CastLayer()(inp)
        preprocessed.append(float_value)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(stack_dict(dict(numeric_features)))

    # Process numeric features
    numeric_inputs = {}
    for name in numeric_feature_names:
        numeric_inputs[name] = inputs[name]

    numeric_inputs = stack_dict(numeric_inputs)
    numeric_normalized = normalizer(numeric_inputs)

    preprocessed.append(numeric_normalized)

    # Process categorical features
    for name in categorical_feature_names:
        vocab = sorted(set(df[name]))
        print(f"name: {name}")
        print(f"vocab: {vocab}\n")

        if isinstance(vocab[0], str):
            lookup = tf.keras.layers.StringLookup(
                vocabulary=vocab, output_mode="one_hot"
            )
        else:
            lookup = tf.keras.layers.IntegerLookup(
                vocabulary=vocab, output_mode="one_hot"
            )

        x = inputs[name][:, tf.newaxis]
        x = lookup(x)
        preprocessed.append(x)

    # Concatenate all tensors
    preprocesssed_result = MyConcatLayer()(preprocessed)

    preprocessor = tf.keras.Model(inputs, preprocesssed_result)

    # Create the model
    body = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    x = preprocessor(inputs)
    result = body(x)

    model = tf.keras.Model(inputs, result)

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    ds = tf.data.Dataset.from_tensor_slices((dict(df), target))
    ds = ds.batch(BATCH_SIZE)
    model.fit(ds, epochs=1)

    return model.predict(ds.take(1))


class CastLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CastLayer, self).__init__(**kwargs)

    def call(self, inp):
        return tf.cast(inp, tf.float32)


class MyConcatLayer(tf.keras.layers.Layer):
    def call(self, values):
        values = [tf.cast(v, tf.float32) for v in values]
        return tf.concat(values, axis=-1)


@pytest.mark.xfail(reason="ValueError: Invalid dtype: object")
def test_full_example_train_with_df(df, target):
    # https://www.tensorflow.org/tutorials/load_data/pandas_dataframe#full_example
    # Inputs are directly passed as dictionary of series

    # ensure deterministic results
    tf.keras.utils.set_random_seed(42)

    numeric_feature_names = ["age", "thalach", "trestbps", "chol", "oldpeak"]
    binary_feature_names = ["sex", "fbs", "exang"]
    categorical_feature_names = ["cp", "restecg", "slope", "thal", "ca"]

    numeric_features = df[numeric_feature_names]

    inputs = {}

    for name, column in df.items():
        if isinstance(column[0], str):
            dtype = tf.string
        elif name in categorical_feature_names or name in binary_feature_names:
            dtype = tf.int64
        else:
            dtype = tf.float32

        inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

    preprocessed = []

    # Process binary features
    for name in binary_feature_names:
        inp = inputs[name]
        inp = inp[:, tf.newaxis]
        float_value = CastLayer()(inp)
        preprocessed.append(float_value)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(stack_dict(dict(numeric_features)))

    # Process numeric features
    numeric_inputs = {}
    for name in numeric_feature_names:
        numeric_inputs[name] = inputs[name]

    numeric_inputs = stack_dict(numeric_inputs)
    numeric_normalized = normalizer(numeric_inputs)

    preprocessed.append(numeric_normalized)

    # Process categorical features
    for name in categorical_feature_names:
        vocab = sorted(set(df[name]))
        print(f"name: {name}")
        print(f"vocab: {vocab}\n")

        if isinstance(vocab[0], str):
            lookup = tf.keras.layers.StringLookup(
                vocabulary=vocab, output_mode="one_hot"
            )
        else:
            lookup = tf.keras.layers.IntegerLookup(
                vocabulary=vocab, output_mode="one_hot"
            )

        x = inputs[name][:, tf.newaxis]
        x = lookup(x)
        preprocessed.append(x)

    # Concatenate all tensors
    preprocesssed_result = MyConcatLayer()(preprocessed)

    preprocessor = tf.keras.Model(inputs, preprocesssed_result)

    # Create the model
    body = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    x = preprocessor(inputs)
    result = body(x)

    model = tf.keras.Model(inputs, result)

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(dict(df), target, epochs=1, batch_size=BATCH_SIZE)

    return model.predict(dict(df[:BATCH_SIZE]))
