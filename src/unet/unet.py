from typing import Optional, Union, Callable, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model#, Input
from tensorflow.keras.layers import Input
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam

import tensorflow_addons as tfa

import metrics as unet_metrics


import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '..', '..', '..', '..' )) # dev/root folder
sys.path.append(os.path.join(dir_path, '..', '..', '..', '..', 'sib_proto', '3D', 'rgbd_semantic_segmentation'))

import sib_rgbd_dataset
import nyu_dataset


physical_devices = tf.config.list_physical_devices('GPU') 
print(f'physical devices: {physical_devices}')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)



class ConvBlock(layers.Layer):

    def __init__(self, layer_idx, filters_root, kernel_size, dropout_rate, padding, activation, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.dropout_rate=dropout_rate
        self.padding=padding
        self.activation=activation
        self.trainable = True

        filters = _get_filter_count(layer_idx, filters_root)
        self.conv2d_1 = layers.Conv2D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size),
                                      kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                      strides=1,
                                      padding=padding)

        #self.batch_norm_1 = layers.BatchNormalization() # vn add
        self.dropout_1 = layers.Dropout(rate=dropout_rate)
        self.activation_1 = layers.Activation(activation)

        self.conv2d_2 = layers.Conv2D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size),
                                      kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                      strides=1,
                                      padding=padding)
        #self.batch_norm_2 = layers.BatchNormalization() # vn add
        self.dropout_2 = layers.Dropout(rate=dropout_rate)

        self.activation_2 = layers.Activation(activation)

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        x = self.conv2d_1(x)
        #x = self.batch_norm_1(x) # vn add

        if training:
            x = self.dropout_1(x)
        x = self.activation_1(x)
        x = self.conv2d_2(x)
        #x = self.batch_norm_2(x) # vn add
        if training:
            x = self.dropout_2(x)

        x = self.activation_2(x)
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    dropout_rate=self.dropout_rate,
                    padding=self.padding,
                    activation=self.activation,
                    **super(ConvBlock, self).get_config(),
                    )


class UpconvBlock(layers.Layer):

    def __init__(self, layer_idx, filters_root, kernel_size, pool_size, padding, activation, **kwargs):
        super(UpconvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        self.padding=padding
        self.activation=activation
        self.trainable = True

        filters = _get_filter_count(layer_idx + 1, filters_root)
        self.upconv = layers.Conv2DTranspose(filters // 2,
                                             kernel_size=(pool_size, pool_size),
                                             kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                             strides=pool_size, padding=padding)

        self.activation_1 = layers.Activation(activation)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.upconv(x)
        x = self.activation_1(x)
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    pool_size=self.pool_size,
                    padding=self.padding,
                    activation=self.activation,
                    **super(UpconvBlock, self).get_config(),
                    )

class CropConcatBlock(layers.Layer):

    def call(self, x, down_layer, **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:,
                                        height_diff: (x2_shape[1] + height_diff),
                                        width_diff: (x2_shape[2] + width_diff),
                                        :]

        x = tf.concat([down_layer_cropped, x], axis=-1)
        return x


def build_model(nx: Optional[int] = None,
                ny: Optional[int] = None,
                channels: int = 1,
                num_classes: int = 2,
                layer_depth: int = 5,
                filters_root: int = 64,
                kernel_size: int = 3,
                pool_size: int = 2,
                dropout_rate: int = 0.0, #0.5,
                padding:str="valid",
                activation:Union[str, Callable]="relu") -> Model:
    """
    Constructs a U-Net model

    :param nx: (Optional) image size on x-axis
    :param ny: (Optional) image size on y-axis
    :param channels: number of channels of the input tensors
    :param num_classes: number of classes
    :param layer_depth: total depth of unet
    :param filters_root: number of filters in top unet layer
    :param kernel_size: size of convolutional layers
    :param pool_size: size of maxplool layers
    :param dropout_rate: rate of dropout
    :param padding: padding to be used in convolutions
    :param activation: activation to be used

    :return: A TF Keras model
    """

    inputs = Input(shape=(ny, nx, channels), name="inputs")

    x = inputs
    contracting_layers = {}

    conv_params = dict(filters_root=filters_root,
                       kernel_size=kernel_size,
                       dropout_rate=dropout_rate,
                       padding=padding,
                       activation=activation)

    for layer_idx in range(0, layer_depth - 1):
        x = ConvBlock(layer_idx, **conv_params)(x)
        contracting_layers[layer_idx] = x
        x = layers.MaxPooling2D((pool_size, pool_size))(x)

    x = ConvBlock(layer_idx + 1, **conv_params)(x)

    for layer_idx in range(layer_idx, -1, -1):
        x = UpconvBlock(layer_idx,
                        filters_root,
                        kernel_size,
                        pool_size,
                        padding,
                        activation)(x)
        x = CropConcatBlock()(x, contracting_layers[layer_idx])
        x = ConvBlock(layer_idx, **conv_params)(x)

    x = layers.Conv2D(filters=num_classes, # Set name on this layer - finetuning
                      kernel_size=(1, 1),
                      kernel_initializer=_get_kernel_initializer(filters_root, kernel_size),
                      strides=1,
                      padding=padding)(x)

    x = layers.Activation(activation)(x) # Set name on this layer - finetuning
    outputs = layers.Activation("softmax", name="outputs")(x)
    model = Model(inputs, outputs, name="unet")

    return model


def _get_filter_count(layer_idx, filters_root):
    return 2 ** layer_idx * filters_root


def _get_kernel_initializer(filters, kernel_size):
    stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
    return TruncatedNormal(stddev=stddev)


def finalize_model(model: Model,
                   loss: Optional[Union[Callable, str]]=losses.categorical_crossentropy,
                   optimizer: Optional= None,
                   metrics:Optional[List[Union[Callable,str]]]=None,
                   dice_coefficient: bool=True,
                   auc: bool=True,
                   mean_iou: bool=True,
                   **opt_kwargs):
    """
    Configures the model for training by setting, loss, optimzer, and tracked metrics

    :param model: the model to compile
    :param loss: the loss to be optimized. Defaults to `categorical_crossentropy`
    :param optimizer: the optimizer to use. Defaults to `Adam`
    :param metrics: List of metrics to track. Is extended by `crossentropy` and `accuracy`
    :param dice_coefficient: Flag if the dice coefficient metric should be tracked
    :param auc: Flag if the area under the curve metric should be tracked
    :param mean_iou: Flag if the mean over intersection over union metric should be tracked
    :param opt_kwargs: key word arguments passed to default optimizer (Adam), e.g. learning rate
    """

    if optimizer is None:
        optimizer = Adam(**opt_kwargs)

    # if metrics is None:
    #     metrics = ['categorical_crossentropy',
    #                'categorical_accuracy',
    #                ]

    metrics=["mae", "acc"]

    # if mean_iou:
    #     metrics += [unet_metrics.mean_iou]

    # if dice_coefficient:
    #     metrics += [unet_metrics.dice_coefficient]

    # if auc:
    #     metrics += [tf.keras.metrics.AUC()]

    model.trainable = True
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics,
                  )




def fit_model(model, output_model_path, training_dataset, validation_dataset, batch_size=2, nbr_epochs=100, load_weights=''):


    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)

    checkpoints_dir = os.path.join(output_model_path, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    checkpoint_max_val_acc_filepath = os.path.join(checkpoints_dir, 'ckpt-max_val_acc.h5')
    checkpoint_min_train_loss_filepath = os.path.join(checkpoints_dir, 'ckpt-min_train_loss.h5')

    # Setup model checkpoint saving
    model_checkpoint_callback_max_val_acc = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_max_val_acc_filepath,
        save_weights_only=True,
        monitor='val_acc',
        mode='max',
        save_best_only=True)
    
    model_checkpoint_callback_min_train_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_min_train_loss_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(output_model_path, 'logs'),
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        write_steps_per_second=False,
        update_freq='epoch',
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None)

    if load_weights != '':
        print(f'loading weights: {load_weights}')
        #model.load_weights(load_weights, by_name=True, skip_mismatch=True) # Weights must be in h5/keras format to load by_name and skip_mismatch ( If tf weights, simply load a model and save as .h5 and re-run)
        model.load_weights(load_weights)

    model.fit(training_dataset, epochs=nbr_epochs, verbose=2, batch_size=batch_size, validation_data=validation_dataset, callbacks=[model_checkpoint_callback_max_val_acc, model_checkpoint_callback_min_train_loss, tensorboard_callback])

    model.save_weights(os.path.join(output_model_path, 'unet_sib_dataset_done_weights'))
    print(f'saving model')
    model.save(os.path.join(output_model_path, 'unet_sib_dataset_done.h5'))


def get_sib_datasets(sample_input_shape, train_base_dir, validation_base_dir):
    
    # include_classes_and_softmax_index = [{'class_name': 'container', 'softmax_index': 1},
    #                                     {'class_name': 'contents', 'softmax_index': 2}]


    include_classes_and_softmax_index = [{'class_name': 'container', 'softmax_index': 1}]

    n_classes = len(include_classes_and_softmax_index) + 1 # +1 if use_zero_as_background_class=True 
    
    train_dataset = sib_rgbd_dataset.SiBRGBDDataset(train_base_dir)
    train_dataset.load_dataset_info()    
    #depth_mean, depth_std, rgb_mean, rgb_std = train_dataset.calculate_dataset_mean_std()


    # Set means to 0 to get positive numbers instead of zero center ( Try fix unet doesnt learn....)
    # depth_mean = 0
    # depth_std = 1.0
    # rgb_mean[0] = 0
    # rgb_mean[1] = 0
    # rgb_mean[2] = 0
    # rgb_std[0] = 1.0
    # rgb_std[1] = 1.0
    # rgb_std[2] = 1.0

    # Set same mean and std for depth as in nyu for finetuning
    depth_mean = nyu_dataset.NyuDepthv2Dataset.dataset_mean_depth # mean ( depth channel ) calculated over the whole nyuv2 dataset
    depth_std = nyu_dataset.NyuDepthv2Dataset.dataset_std_depth
    rgb_mean = nyu_dataset.NyuDepthv2Dataset.dataset_mean_colors
    rgb_std = nyu_dataset.NyuDepthv2Dataset.dataset_std_colors

    train_dataset.set_depth_normalize_mean_std_params(depth_mean, depth_std)
    train_dataset.set_rgb_normalize_mean_std_params(rgb_mean, rgb_std)
    train_dataset.set_include_classes_and_softmax_index(include_classes_and_softmax_index, use_zero_as_background_class=True)
    train_dataset.set_shuffle(True)
    

    val_dataset = sib_rgbd_dataset.SiBRGBDDataset(validation_base_dir)
    val_dataset.load_dataset_info()
    val_dataset.set_depth_normalize_mean_std_params(depth_mean, depth_std)
    val_dataset.set_rgb_normalize_mean_std_params(rgb_mean, rgb_std)
    val_dataset.set_include_classes_and_softmax_index(include_classes_and_softmax_index, use_zero_as_background_class=True)
    

    tf_output_shape = tf.TensorShape([None, sample_input_shape[0], sample_input_shape[1], n_classes])
    dt = tf.data.Dataset.from_generator(train_dataset.__next__,  (tf.float32, tf.float32), (tf.TensorShape([None, sample_input_shape[0], sample_input_shape[1], sample_input_shape[2]]), tf_output_shape))
    dv = tf.data.Dataset.from_generator(val_dataset.__next__,  (tf.float32, tf.float32), (tf.TensorShape([None, sample_input_shape[0], sample_input_shape[1], sample_input_shape[2]]), tf_output_shape))

    return dt, dv, train_dataset, val_dataset, n_classes


def set_finetune_only(model):

    for layer in model.layers:
        if layer.name in ['conv2d_14', 'activation_17', 'outputs']:
            layer.trainable = True
        else:
            layer.trainable = False

        print(f'Name: {layer.name}, {layer}')    


def get_nyu_dataset(dataset_file_path, use_zero_as_background_class=False, set_use_pixel_augmentation=False, use_spatial_augmentation=False):

    # composite_include_classes = [['wall', 'blinds', 'wall decoration', 'wall divider', 'whiteboard'], # TODO reduce this class since it becomes to heavy 
    #     ['floor', 'floor mat', 'rug'],
    #     'ceiling']


    input_classes = ['wall', 'floor', 'desk', 'cup', 'bookshelf', 'clothes', 'shelves', 'blinds', 'books', 'book', 'sofa', 'bed', 'counter',
    'bag', 'lamp', 'box', 'paper', 'ceiling', 'bottle', 'pillow', 'door', 'window', 'table', 'chair', 'cabinet', 'picture']


    sample_input_shape = (480,640,4)
    n_classes = len(input_classes)
    if use_zero_as_background_class:
        n_classes += 1
    
    flatten_outputs = False
    train_dataset = nyu_dataset.NyuDepthv2Dataset(dataset_file_path, 
                                sample_input_shape = sample_input_shape,
                                train_split = 0.9, 
                                split='train',
                                normalize_depth_strategy=nyu_dataset.NormalizationStrategy.NYU_DATASET_DEPTH_MEAN_STD_NEG1_POS1_RGB_NEG1_POS1, 
                                flatten_outputs=flatten_outputs,
                                shuffle=True)  # Sample input shape is RGBD (cols, rows, 4channel=rgbd)

    train_dataset.set_include_classes(input_classes, use_zero_as_background_class)
    if set_use_pixel_augmentation:
        print("Enabling pixel augmentation for training data")
        train_dataset.set_use_pixel_augmentation()

    if use_spatial_augmentation:
        print("Enabling spatial augmentation for training data")
        train_dataset.set_use_spatial_augmentation()


    validation_dataset = nyu_dataset.NyuDepthv2Dataset(dataset_file_path, 
                            sample_input_shape = sample_input_shape,
                            train_split = 0.9, 
                            split='validation',
                            normalize_depth_strategy=nyu_dataset.NormalizationStrategy.NYU_DATASET_DEPTH_MEAN_STD_NEG1_POS1_RGB_NEG1_POS1, 
                            flatten_outputs=flatten_outputs,
                            shuffle=False)  # Sample input shape is RGBD (cols, rows, 4channel=rgbd)

    validation_dataset.set_include_classes(input_classes, use_zero_as_background_class)

    tf_output_shape = tf.TensorShape([None, sample_input_shape[0], sample_input_shape[1], n_classes])
    dt = tf.data.Dataset.from_generator(train_dataset.__next__,  (tf.float32, tf.float32), (tf.TensorShape([None, sample_input_shape[0], sample_input_shape[1], sample_input_shape[2]]), tf_output_shape))
    dv = tf.data.Dataset.from_generator(validation_dataset.__next__,  (tf.float32, tf.float32), (tf.TensorShape([None, sample_input_shape[0], sample_input_shape[1], sample_input_shape[2]]), tf_output_shape))

    return dt, dv, train_dataset, validation_dataset, n_classes

def create_model_info(output_model_path, layer_depth, loss, normalization, classes):

    classes_file_path = os.path.join(output_model_path, 'classes.txt')
    
    # Write classes.txt file
    with open(classes_file_path, 'w+') as classes_f:
        for c in classes:
            classes_f.write(f'{c}\n')

    info_file_path = os.path.join(output_model_path, 'info.txt') 

    # write info file
    with open(info_file_path, 'w+') as info_f:
        pass



if __name__ == "__main__":

    sample_input_shape = (480,640,4)
    
    #output_model_path = '/home/viktor/ml/rgbd_unet/unet_depth_4_sibdataset_220420'

    nyu_path = '/home/viktor/datasets/SOURCE_DATASETS/rgbd/nyu_depth_v2_labeled.mat'
    #nyu_path = 'C:\\datasets\\SOURCE_DATASETS\\rgbd\\nyu_depth_v2_labeled.mat'

    tf_weigths = '/home/viktor/ml/rgbd_unet/unet_depth_5_nyu_neg1pos1norm_220421/checkpoints/ckpt-min_train_loss'
    #tf_weigths = 'C:\\ml\\rgbd_unet\\unet_depth_4_nyu_24_classes_20420\\ckpt-max_val_acc_epoch_306'
    h5_weights = tf_weigths + '.h5'

    if False:
        
        # Convert tf weights to hd5 weights        
        model = build_model(nx=640, ny=480, channels=4, layer_depth=5, num_classes=24, padding='same')
        model.summary()
        model.load_weights(tf_weigths)
        model.save_weights(h5_weights)
        exit(0)

    if True:

        ### NYU training ####

        #h5_weights = '/home/viktor/ml/rgbd_unet/unet_depth_5_nyu_26_classesneg1pos1norm_focal_loss_220422/checkpoints/ckpt-min_train_loss.h5'
        output_model_path = '/home/viktor/ml/rgbd_unet/unet_depth_5_nyu_27_classesneg1pos1norm_focal_loss_spat_aug_bgclass_220428'
        
        dt, dv, train_dataset, val_dataset, n_classes = get_nyu_dataset(nyu_path, use_zero_as_background_class=True, set_use_pixel_augmentation=False, use_spatial_augmentation=True)
        #train_dataset.visualize_data_set()
        
        model = build_model(nx=640, ny=480, channels=4, layer_depth=5, num_classes=n_classes, padding='same')
        model.summary()
        #set_finetune_only(model)

        optimizer = Adam(learning_rate=0.001)
        loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.5)

        finalize_model(model, loss=loss, optimizer=optimizer)
        
        #model.load_weights(h5_weights)#, by_name=True, skip_mismatch=True)
        fit_model(model, output_model_path, dt, dv, batch_size=24, nbr_epochs=1000, load_weights='')
        exit(0)

    if False:

        validation_base_dir = '/home/viktor/datasets/GENERATED_DATA_SETS/rgbd/vn_large_container_composit_validation_220427'
        train_base_dir = '/home/viktor/datasets/GENERATED_DATA_SETS/rgbd/vn_large_container_composit_training_220427'
        #validation_base_dir = '/home/viktor/datasets/RAW_DATA/stereo_large_container/vn_office'

        output_model_path = '/home/viktor/ml/rgbd_unet/unet_depth_5_nyu_finetune_focal_sib_neg1pos1norm_inclbgclass_container_220428'
        dt, dv, train_dataset, val_dataset, n_classes = get_sib_datasets(sample_input_shape, train_base_dir, validation_base_dir)
        #train_dataset.visualize_data_set()
        
        
        h5_weights = '/home/viktor/ml/rgbd_unet/unet_depth_5_nyu_26_classesneg1pos1norm_focal_loss_220422/checkpoints/ckpt-min_train_loss.h5'
        model = build_model(nx=640, ny=480, channels=4, layer_depth=5, num_classes=n_classes, padding='same')
        model.summary()
        set_finetune_only(model)

        optimizer = Adam(learning_rate=0.0001)
        loss = tfa.losses.SigmoidFocalCrossEntropy()
        finalize_model(model, loss=loss, optimizer=optimizer)
        
        model.load_weights(h5_weights, by_name=True, skip_mismatch=True)
        fit_model(model, output_model_path, dt, dv, batch_size=24, nbr_epochs=5000, load_weights='') # High batch size seems important for  segmentation
        exit(0)


    if False:
        import rgbd_sem_seg_dev_app_main
        input_image_folder = '/home/viktor/datasets/RAW_DATA/stereo_large_container/vn_office'
        weights_path = '/home/viktor/ml/rgbd_unet/unet_depth_4_sib_dataset_220414/unet_sib_dataset_done_weights'
        app = rgbd_sem_seg_dev_app_main.RGBDSegApp()
        model.load_weights(weights_path)
        app.model = model
        app.classes = ['container', 'contents']
        app.flat_output = False
        app.process_single_frames(input_image_folder, model)

    



    pass