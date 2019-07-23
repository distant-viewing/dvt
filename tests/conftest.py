import pytest


@pytest.fixture(scope="session")
def setup_tensorflow():
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

    # fix a local bug
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
