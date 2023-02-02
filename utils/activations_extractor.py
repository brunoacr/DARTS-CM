import dill
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model

from os import PathLike
from pathlib import Path
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

# filename of instance when saved
_FILENAME = 'activations_extractor.pkl'
# extractor model folder name when saved
_MODEL_FOLDER_NAME = 'extractor_model'


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
    elif axes_to_keep < len(tensor.shape) - 1:
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


class ActivationsExtractor:
    """Class representing an object capable of extracting activations from a
     set of layers and/or units from a neural network."""

    def __init__(self, model, to_extract,
                 behavior=None,
                 concatenate_outputs=True,
                 extracting_model_name='link_to_main_nn'):
        """
        Initializes an object representing a neural network model activation
         extractor.

        Parameters
        ----------
        model : Model
            A neural network model.
        to_extract : {dict, list[str], list[int]}
        Either a dictionary with layer (index or name) as key and a list of the
         indexes of units from that layer whose activations are to be
         extracted, or a list of layers (indexes or names) from which all
        activations will be extracted.
        behavior : str, optional
            String representing the expected behavior of the network model,
            used to deduce the expected activations' shape.
        concatenate_outputs : bool, optional
            Whether to attempt to concatenate the activations extracted from
             different layers. Note: The concatenation only occurs if the
             extracted activation shapes are compatible.
        extracting_model_name : str, optional
            Name of the model used to extract the activations.
        """
        self.to_extract = to_extract
        self.behavior = behavior
        self.concatenate_outputs = concatenate_outputs
        self.model = get_extracting_model(
            main_model=model,
            extracting=self.to_extract,
            behavior=self.behavior,
            concatenate_outputs=self.concatenate_outputs,
            extracting_model_name=extracting_model_name
        )

        self.model_path = None
        self.model_loader = None

    def __getstate__(self):
        """
        Determines what needs to be serialized.

        Returns
        -------
        dict
            Object representing the instance's state.
        """
        return {
            'to_extract': self.to_extract,
            'behavior': self.behavior,
            'concatenate_outputs': self.concatenate_outputs,
            'model_path': self.model_path,
            'model_loader': self.model_loader,
        }

    def __setstate__(self, state):
        """
        Given a serialized state, reconfigures the instance.
        
        Parameters
        ----------
        state
            Object representing the instance's state.
        """
        self.to_extract = state['to_extract']
        self.behavior = state['behavior']
        self.concatenate_outputs = state['concatenate_outputs']
        self.model = None
        self.model_path = state['model_path']
        self.model_loader = state['model_loader']

    def extract_activations(self, x,
                            batch_size=None, verbose=0, steps=None,
                            callbacks=None, max_queue_size=10, workers=1,
                            use_multiprocessing=False):
        """
        Given a set of samples, returns a numpy array/pandas dataframe of the
         layers activations.

        Parameters
        ----------
        x
            Input samples.
        batch_size : int or `None`, optional
            Number of samples per batch.
        verbose : 0 or 1, optional
            Verbosity mode.
        steps : optional
            Total number of steps (batches of samples) before declaring the
             prediction round finished.
        callbacks : List of `keras.callbacks.Callback` instances, optional
            List of callbacks to apply during prediction.
        max_queue_size : int, optional
            Used for generator or `keras.utils.Sequence` input only. Maximum
             size for the generator queue.
        workers : int, optional
            Used for generator or `keras.utils.Sequence` input only. Maximum
             number of processes to spin up when using process-based threading.
        use_multiprocessing : bool, optional
            Used for generator or `keras.utils.Sequence` input only. If `True`,
             use process-based threading. If unspecified, `use_multiprocessing`
             will default to `False`.

        Returns
        -------
        Any
            Numpy array(s) of predictions containing the extracted activations.

        Raises
        -------
        RuntimeError: If `model.predict` is wrapped in `tf.function`.
        ValueError: In case of mismatch between the provided input data and the
         model's expectations, or in case a stateful model receives a number of
         samples that is not a multiple of the batch size.
        """
        return self.model.predict(x,
                                  batch_size=batch_size,
                                  verbose=verbose,
                                  steps=steps,
                                  callbacks=callbacks,
                                  max_queue_size=max_queue_size,
                                  workers=workers,
                                  use_multiprocessing=use_multiprocessing)

    def save(self, path,
             model_path=None,
             model_loader=load_model):
        """
        Saves the ActivationsExtractor instance to a single pickle (dill) file.
         If a path is given for the extractor's model, it is assumed to already
         be saved on that path, if no path is provided, the model is saved
         separately.
        Note: A warning regarding the extractor's model compilation may be
         produced. This is normal behavior, since this model does not require
         compilation as it will not be trained.

        Parameters
        ----------
        path : str, Path, PathLike
            A path to the folder to save the activations extractor.
        model_path : str, Path, PathLike, optional
            A path to where the extractor model is saved. If a path is passed
             the extractor model is assumed to be saved there, else the
             extractor model will be saved under `path`.
        model_loader : callable, optional
            Function to load the extractor model given its path. Useful if the
             extractor model is a custom model, i.e., not loadable using the
             default `tensorflow.keras.models.load_model` function.
        """
        if model_path is None:  # save extractor model to default path
            self.model.save(Path(path, _MODEL_FOLDER_NAME))
        else:
            self.model_path = Path(model_path)

        self.model_loader = model_loader

        with open(Path(path, _FILENAME), 'wb') as file:
            dill.dump(self, file)

    @staticmethod
    def load(path, extracting_model=None):
        """
        Loads an ActivationsExtractor instance from a single pickle (dill)
         file.
        Note: A warning regarding the extractor's model compilation may be
         produced. This is normal behavior, since this model does not require
         compilation as it will not be trained.

        Parameters
        ----------
        path : str, Path, PathLike
            A path to the folder to load the activations extractor.
        extracting_model : Model, optional
            The activations extractor's model.

        Returns
        -------
        ActivationsExtractor
            An activation extractor instance.
        """
        with open(Path(path, _FILENAME), 'rb') as file:
            activations_extractor = dill.load(file)

        if extracting_model:
            activations_extractor.model = extracting_model
        else:  # only attempts to load the model if no model was provided
            model_path = activations_extractor.model_path if activations_extractor.model_path is not None else Path(
                path, _MODEL_FOLDER_NAME)
            activations_extractor.model = activations_extractor.model_loader(model_path)
        return activations_extractor
