import tensorflow as tf
import tflearn

def construct_dnn():
    tf.reset_default_graph()
    tflearn.init_graph(num_cores=4, gpu_memory_fraction=0.3)
    tflearn.config.init_training_mode()
    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_90degrees_rotation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_flip_updown()
    input_layer = tflearn.input_data(shape=[None, 15, 15, 3], data_augmentation=img_aug)
    # block 1
    net = tflearn.conv_2d(input_layer, 256, 3, activation=None)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, activation='relu')
    # res block 1
    tmp = tflearn.conv_2d(net, 256, 3, activation=None)
    tmp = tflearn.batch_normalization(tmp)
    tmp = tflearn.activation(tmp, activation='relu')
    tmp = tflearn.conv_2d(tmp, 256, 3, activation=None)
    tmp = tflearn.batch_normalization(tmp)
    net = tflearn.activation(net + tmp, activation='relu')
    # res block 2
    tmp = tflearn.conv_2d(net, 256, 3, activation=None)
    tmp = tflearn.batch_normalization(tmp)
    tmp = tflearn.activation(tmp, activation='relu')
    tmp = tflearn.conv_2d(tmp, 256, 3, activation=None)
    tmp = tflearn.batch_normalization(tmp)
    net = tflearn.activation(net + tmp, activation='relu')
    # res block 3
    tmp = tflearn.conv_2d(net, 256, 3, activation=None)
    tmp = tflearn.batch_normalization(tmp)
    tmp = tflearn.activation(tmp, activation='relu')
    tmp = tflearn.conv_2d(tmp, 256, 3, activation=None)
    tmp = tflearn.batch_normalization(tmp)
    net = tflearn.activation(net + tmp, activation='relu')
    # res block 4
    tmp = tflearn.conv_2d(net, 256, 3, activation=None)
    tmp = tflearn.batch_normalization(tmp)
    tmp = tflearn.activation(tmp, activation='relu')
    tmp = tflearn.conv_2d(tmp, 256, 3, activation=None)
    tmp = tflearn.batch_normalization(tmp)
    net = tflearn.activation(net + tmp, activation='relu')
    # value head
    net = tflearn.conv_2d(net, 1, 1, activation=None)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, activation='relu')
    net = tflearn.fully_connected(net, 256, activation='relu')
    final = tflearn.fully_connected(net, 1, activation='tanh')
    # optmizer
    #sgd = tflearn.optimizers.SGD(learning_rate=0.01, lr_decay=0.95, decay_step=200000)
    sgd = tflearn.optimizers.SGD(learning_rate=0.0001, lr_decay=0.95, decay_step=1000000)
    regression = tflearn.regression(final, optimizer=sgd, loss='mean_square',  metric='R2')
    model = tflearn.DNN(regression)
    return model
