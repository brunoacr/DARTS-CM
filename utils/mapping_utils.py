import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model


def _flatten_tensor_axes(tensor, axes_to_keep):
    """
    Partially flattens a tensor, keeping the first `axes_to_keep` axes
     unaltered.

    Parameters
    ----------
    tensor : Tensor
        The tensor to be flattened.
    axes_to_keep : int
        The amount of axes to be kept unchanged.

    Returns
    -------
    tf.Tensor
        The flattened tensor.
    """
    if axes_to_keep <= 0:  # flatten the whole tensor
        return tf.reshape(tensor, [-1])
    elif axes_to_keep < len(tensor.shape)-1:
        # keep the first `axes_to_keep` axes, flatten the remaining axes
        kept_dims = tf.shape(tensor)[:axes_to_keep]
        flatten_dims = tf.reduce_prod(tf.shape(tensor)[axes_to_keep:], keepdims=True)
        return tf.reshape(tensor, tf.concat([kept_dims, flatten_dims], axis=0))
    else:  # axes_to_keep >= len(tensor.shape) - 1
        return tensor


def get_extracting_model(main_model, extracting,
                         behavior=None,
                         concatenate_outputs=True,
                         extracting_model_name='link_to_main_nn'):
    """
    Creates a model that returns the specified activations of another model.

    Parameters
    ----------
    main_model : Model
        The neural network model from which the activations will be extracted.
    extracting : {dict, list[str], list[int]}
        Either a dictionary with layer (index or name) as key and a list of the
         indexes of units from that layer whose activations are to be
         extracted, or a list of layers (indexes or names) from which all
        activations will be extracted.
    behavior : {None, 'flatten_feedforward', 'flatten_recurrent', dict}, optional
        Either `None' to specify the output of the model to be the (unaltered)
         numpy array(s) of predictions containing the extracted activations, or
         'flatten_feedforward' for returning all extracted activations in a
         single numpy array preserving only the first axis and flattening the
         remaining, i.e., (samples, #activations), or 'flatten_recurrent' for
         returning all extracted activations in single numpy array preserving
         only the first two axes and flattening the remaining, i.e.,
         (samples, timesteps, #activations), or a dictionary whose keys are
         layers (indexes or names) and values are integers specifying the
         amount of dimensions to be kept in activations extracted from that
         layer (flattens the remaining dimensions).
    concatenate_outputs : bool, optional
            Whether to attempt to concatenate the activations extracted from
             different layers. Note: The concatenation only occurs if the
             extracted activation shapes are compatible.
    extracting_model_name : str, optional
        Name of the model to be created.

    Returns
    -------
    Model
        Created model, which input is the same as the specified `main_model`
        and whose outputs are the specified.
    """
    # start by getting the output tensors of each extracted layer
    extracted_tensors = [
        main_model.get_layer(name=layer_id).output if isinstance(layer_id, str)
        else main_model.get_layer(index=layer_id).output
        for layer_id in extracting
    ]

    # `behavior` indicates the amount of axis to be kept for each layer
    if behavior is not None:
        # `flatten_feedforward` behavior preserves the shape of the first axis,
        # flattening the remaining
        if behavior == 'flatten_feedforward':
            _behavior = [1 for _ in extracting]
        # `flatten_recurrent` behavior preserves the shape of the first two
        # axes, flattening the remaining
        elif behavior == 'flatten_recurrent':
            _behavior = [2 for _ in extracting]
        elif isinstance(behavior, dict):
            # `behavior` is kept in the same order as `extracting`
            _behavior = [behavior[layer_id] for layer_id in extracting]

        # apply the specified behavior
        extracted_tensors = [
            tensor if tensor.get_shape().rank <= axes_to_keep + 1
            else _flatten_tensor_axes(tensor, axes_to_keep)
            for (tensor, axes_to_keep) in zip(extracted_tensors, _behavior)
        ]

    if isinstance(extracting, dict):
        # extracting from a dict indicating for each layer, which neurons
        # should be extracted
        neuron_sets = [extracting[layer] for layer in extracting
                       if extracting[layer] is not None
                       and len(extracting[layer]) > 0]

        if behavior is None:
            # default behavior is to extract the activations from the last axis
            _behavior = [-1 for _ in extracting]

        extracted_tensors = [
            tf.gather(tensor, indices=list(neurons), axis=axes_to_keep)
            for (tensor, neurons, axes_to_keep)
            in zip(extracted_tensors, neuron_sets, _behavior)
        ]

    if concatenate_outputs and len(extracted_tensors) > 1:
        # check if all outputs share the same shape (except for the last axis)
        output_shapes = set([tuple(tensor.shape[:-1])
                             for tensor
                             in extracted_tensors])
        if len(output_shapes) == 1:
            extracted_tensors = tf.concat(extracted_tensors, axis=-1)

    extracting_model = Model(inputs=main_model.input,
                             outputs=extracted_tensors,
                             name=extracting_model_name)
    extracting_model.trainable = False
    return extracting_model
