import time
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


# Define the custom Fixed Dropout layer used in Efficientnet models
class FixedDropout(tf.keras.layers.Dropout):
  def call(self, inputs, training=None):
    return inputs

def get_flops(model, batch_size=None):
    """
    Calculate FLOPS for a model
    """
    if batch_size is None:
        batch_size = 1

    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype))

    # Pass the custom layer class to the custom_objects parameter
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                            run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops

def get_model_paramters(model_path):
    """
    Calculate number of parameters for a model
    """
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        model = tf.keras.models.load_model(model_path, custom_objects={'FixedDropout': FixedDropout})
    
    params = model.count_params()
    return params

def get_inference_speed(model, data):
    """
    Calculate inference speed of a model in ms / image
    """
    #  GPU warm up
    model.predict(data)
    
    start_time = time.time()
    # Get average of 100 inference loops to get an average
    for i in range(100):
        model.predict(data)
    end_time = time.time()
    speed = (end_time - start_time) / (100 * len(data))
    # Return speed in miliseconds
    return speed * 1000