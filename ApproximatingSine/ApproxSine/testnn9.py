import numpy as np
import neuralnetwork_ex9 as nn


# image = np.random.rand(32, 32)
# kernal = np.random.rand(5, 5)





# def subsample_layer(layer, pool_size, strides):
#     layer_rows = ((layer.shape[0] - pool_size) / strides) + 1
#     layer_cols = ((layer.shape[1] - pool_size) / strides) + 1
#     feature_map = np.zeros((layer_rows,layer_cols))
    
#     for row in range(layer_rows):
#         for col in range(layer_cols):
#             start_row = (strides * row)
#             end_row = (start_row + pool_size)
#             start_col = (strides * col)
#             end_col = (start_col + pool_size)
#             feature_unit = np.sum([layer[start_row:end_row, start_col:end_col]])
#             feature_map[row, col] = feature_unit


# inputs
# filters
# kernel_shapes
# padding
# activations

# convnn = newnn(
#     num_maps_layers=[32, 64],
#     kernal_layers=[(5, 5), (5, 5)],
#     kernal_strides=[(1,1), (1,1)],
#     pooling_layers=[(2, 2), (2, 2)],
#     pooling_strides=[(2, 2), (2, 2)],
#     activation_layers=["relu", "relu"],
#     padding_layers=["same", "same"]
# )

# for num_maps in num_maps_layers:
#     for feature_map in range(num_maps):

def compute_activation_fn(num, string):
    num = np.squeeze(num)
    if num < 0:
        return 0
    else:
        return num
def compute_activation_fn_deriv(num, string):
    num = np.squeeze(num)
    if num < 0:
        return 0
    else:
        return 1


def pad_input_layer(input_layer, filter_layer, stride_size, padding_fn):
    """
        ``input_layer`` must be of shape (batch_size, in_height, in_width, in_channels)
        ``filter_layer`` must be of shape (batch_size, filter_height, filter_width, in_channels) or [filter_height, filter_width]
        ``stride_size`` must be of shape [stride_height, stride_width]
        ``padding_fn`` must be a string, "valid" or "same"

        if padding_fn == "valid", returns output of shape that so that kernel and stride can easily stride across and fit
        if padding_fn == "same", returns output padded with 0's so that the feature map will later be the same size as the original input.
    """
    batch_size = input_layer.shape[0]
    padded_input_layer = list()
    for idx in range(batch_size):
        inputs = input_layer[idx]
        input_rows = inputs.shape[0]
        input_cols = inputs.shape[1]

        if type(filter_layer) == list:
            filter_rows = filter_layer[0]
            filter_cols = filter_layer[1]
        else:
            filter_rows = filter_layer[idx].shape[0]
            filter_cols = filter_layer[idx].shape[1]

        stride_rows = stride_size[0]
        stride_cols = stride_size[1]

        if padding_fn == "same":
            width_padding = (input_cols * stride_cols) + filter_cols - input_cols - stride_cols
            height_padding = (input_rows * stride_rows) + filter_rows - input_rows - stride_rows
            row_padding_right = int(np.ceil(width_padding / 2))
            row_padding_left = int(np.floor(width_padding / 2))
            col_padding_bottom = int(np.ceil(height_padding / 2))
            col_padding_top = int(np.floor(height_padding / 2))
            padded_inputs = np.pad(inputs, [(col_padding_top, col_padding_bottom), (row_padding_left, row_padding_right), (0,0)], mode='constant')
        elif padding_fn == "valid":
            max_num_rows = (int)((input_rows - filter_rows) / stride_rows) + 1
            max_num_cols = (int)((input_cols - filter_cols) / stride_cols) + 1
            padded_inputs = inputs[:filter_rows + (stride_rows * (max_num_rows - 1)),:filter_cols + (stride_cols * (max_num_cols - 1))]
        padded_input_layer.append(padded_inputs)
    return np.array(padded_input_layer)

def convert_padded_derivs_layer(orig_input_layer, padded_derivs_layer, filter_layer, stride_size, padding_fn):
    """
        ``orig_input_layer`` must be of shape (batch_size, in_height, in_width, in_channels)
        ``padded_derivs_layer`` must be of shape (batch_size, in_height, in_width, in_channels)
        ``filter_layer`` must be of shape (batch_size, filter_height, filter_width, in_channels)
        ``stride_size`` must be of shape [stride_height, stride_width]
        ``padding_fn`` must be a string, "valid" or "same"

        returns d of padded derivs w.r.t orig_inputs, return shape = (batch_size, orig_input_layer_in_width, orig_input_layer_in_height, orig_input_layer_in_channels)
    """
    batch_size = orig_input_layer.shape[0]
    converted_padded_derivs_layer = list()
    for idx in range(batch_size):
        orig_inputs, padded_derivs = orig_input_layer[idx], padded_derivs_layer[idx]
        input_rows = orig_inputs.shape[0]
        input_cols = orig_inputs.shape[1]

        if type(filter_layer) == list:
            filter_rows = filter_layer[0]
            filter_cols = filter_layer[1]
        else:
            filter_rows = filter_layer[idx].shape[0]
            filter_cols = filter_layer[idx].shape[1]

        stride_rows = stride_size[0]
        stride_cols = stride_size[1]

        if padding_fn == "same":
            width_padding = (input_cols * stride_cols) + filter_cols - input_cols - stride_cols
            height_padding = (input_rows * stride_rows) + filter_rows - input_rows - stride_rows
            row_padding_right = int(np.ceil(width_padding / 2))
            row_padding_left = int(np.floor(width_padding / 2))
            col_padding_bottom = int(np.ceil(height_padding / 2))
            col_padding_top = int(np.floor(height_padding / 2))
            converted_padded_derivs = padded_derivs[col_padding_top:-col_padding_bottom, row_padding_left:-row_padding_right]
        elif padding_fn == "valid":
            max_num_rows = (int)((input_rows - filter_rows) / stride_rows) + 1
            max_num_cols = (int)((input_cols - filter_cols) / stride_cols) + 1
            cut_bottom_rows = input_rows - (filter_rows + (stride_rows * (max_num_rows - 1)))
            cut_right_cols = input_cols - (filter_cols + (stride_cols * (max_num_cols - 1)))
            converted_padded_derivs = np.pad(padded_derivs, [(0, cut_bottom_rows), (0, cut_right_cols), (0, 0)], mode='constant')
        converted_padded_derivs_layer.append(converted_padded_derivs)
    return np.array(converted_padded_derivs_layer)




def generate_fmaps(input_layer, kernel_layer, bias_layer, stride_size, padding_fn, activation_fn, dx=None):
    """
        ``input_layer`` must be of shape (batch_size, in_height, in_width, in_channels)
        ``kernel_layer`` must be of shape (num_filters, filter_height, filter_width, in_channels)
        (in_channels for ``input_layer`` and ``kernel_layer`` must be same size)
        ``bias_layer`` must be of shape (num_filters, 1, 1, 1)

        ``stride_size`` must be of shape [stride_height, stride_width]
        ``padding_fn`` must be a string, "valid" or "same"
        ``activation_fn`` must be a string, e.g. "relu" or "sigmoid"
        ``dx`` is default None, it can be set to "kernel_layer," "bias_layer," or "input_layer" --> then the return will be d output w.r.t dx, of shape (batch_size, dx_height, dx_width, dx_in_channels)


        if ``dx`` is None, returns output of shape (batch_size, fmap_rows, fmap_cols, num_filters)
    """
    batch_size = input_layer.shape[0]
    num_filters = kernel_layer.shape[0]
    kernel_rows = kernel_layer.shape[1]
    kernel_cols = kernel_layer.shape[2]
    
    if padding_fn == "same":
        """ 
            fmap_rows == original input rows
            fmap_cols == original input cols
        """
        fmap_rows = input_layer.shape[1]
        fmap_cols = input_layer.shape[2]
    elif padding_fn == "valid":
        """ 
            fmap_rows == ((in_height - filter_height)/stride_height) + 1
            fmap_cols == ((in_width - filter_width)/stride_width) + 1
        """
        fmap_rows = int(((input_layer.shape[1] - kernel_layer.shape[1]) / stride_size[0]) + 1)
        fmap_cols = int(((input_layer.shape[2] - kernel_layer.shape[2]) / stride_size[1]) + 1)
    
    fmaps_layer = np.zeros((batch_size, fmap_rows, fmap_cols, num_filters))
    
    inputs = pad_input_layer(input_layer, kernel_layer, stride_size, padding_fn)

    if dx == "input_layer":
        dx_layer = np.zeros(inputs.shape)
    elif dx == "kernel_layer":
        dx_layer = np.zeros(kernel_layer.shape)

    for inputs_idx in range(batch_size):
        for kernel_idx in range(num_filters):
            fmap = np.zeros((fmap_rows, fmap_cols))
            for row in range(fmap_rows):
                for col in range(fmap_cols):
                    cur_row = (stride_size[0] * row)
                    cur_col = (stride_size[1] * col)
                    image_section = inputs[inputs_idx, cur_row:(kernel_rows + cur_row), cur_col:(kernel_cols + cur_col)]
                    kernel = kernel_layer[kernel_idx]
                    bias = bias_layer[kernel_idx]
                    featureZ = np.sum(np.multiply(image_section, kernel)) + bias
                    featureA = compute_activation_fn(featureZ, activation_fn)
                    fmap[row, col] = np.squeeze(featureA)

                    if dx == "input_layer":
                        dx_layer[inputs_idx, cur_row:(kernel_rows + cur_row), cur_col:(kernel_cols + cur_col), :] += kernel * compute_activation_fn_deriv(featureZ, activation_fn)
                    elif dx == "kernel_layer":
                        dx_layer[kernel_idx, :, :, :] += image_section * compute_activation_fn_deriv(featureZ, activation_fn)
            fmaps_layer[inputs_idx, :, :, kernel_idx] = fmap
    
    if dx == "input_layer":
        dx_layer = convert_padded_derivs_layer(input_layer, np.array(dx_layer), kernel_layer, [2,2], padding_fn)
        return dx_layer
    elif dx == "kernel_layer":
        return dx_layer
    elif dx == None:
        return fmaps_layer

#print(generate_fmaps(np.random.rand(1,9,9,3), np.random.rand(2,2,2,3), np.random.rand(2,1,1,1), [2,2], "valid", "relu", dx="input_layer"))



def pool_fmaps(input_layer, pool_size, stride_size, pool_fn, padding_fn, dx=None, grad_ys=None):
    """ 
        ``input_layer`` must be of shape (batch_size, in_height, in_width, in_channels)
        ``pool_size`` must be of shape [pool_height, pool_width]
        ``stride_size`` must be of shape [stride_height, stride_width]
        ``pool_fn`` is the pool function used, a string that can be "max," "mean"
        ``padding_fn`` must be a string, "valid" or "same"
        ``dx`` is default None, it can be set to "input_layer"  --> then the return will be d of output w.r.t dx, of shape (batch_size, dx_height, dx_width, dx_in_channels)


        if ``dx`` is None, returns output of shape (batch_size, out_height, out_width, in_channels)
        out_height = ((in_height - pool_height)/stride_height) + 1
        out_width = ((in_width - pool_width)/stride_width) + 1
    """
    batch_size = input_layer.shape[0]
    in_channels = input_layer.shape[3]

    if padding_fn == "same":
        """ 
            pool_rows == original input rows
            pool_cols == original input cols
        """
        pool_rows = input_layer.shape[1]
        pool_cols = input_layer.shape[2]
    elif padding_fn == "valid":
        """ 
            pool_rows == ((in_height - pool_height)/stride_height) + 1
            pool_cols == ((in_width - pool_width)/stride_width) + 1
        """
        pool_rows = int(((input_layer.shape[1] - pool_size[0]) / stride_size[0]) + 1)
        pool_cols = int(((input_layer.shape[2] - pool_size[1]) / stride_size[1]) + 1)
    
    pooled_input_layer = np.zeros((batch_size, pool_rows, pool_cols, in_channels))
    padded_inputs = pad_input_layer(input_layer, pool_size, stride_size, padding_fn)
    dx_layer = np.zeros(padded_inputs.shape)

    for inputs_idx in range(batch_size):
        for channel_idx in range(in_channels):
            inputs = padded_inputs[inputs_idx, :, :, channel_idx]
            pooled_inputs = np.zeros((pool_rows, pool_cols))
            for row in range(pool_rows):
                for col in range(pool_cols):
                    cur_row = (stride_size[0] * row)
                    cur_col = (stride_size[1] * col)
                    features = inputs[cur_row:(pool_size[0] + cur_row), cur_col:(pool_size[1] + cur_col)]
                    if pool_fn == "max":
                        pooled_inputs[row, col] = np.squeeze(np.amax(features))
                        if grad_ys == None:
                            np.put(dx_layer[inputs_idx, cur_row:(pool_size[0] + cur_row), cur_col:(pool_size[1] + cur_col), channel_idx], np.argmax(features), 1)

            pooled_input_layer[inputs_idx,:,:, channel_idx] = pooled_inputs
    if dx == "input_layer":
        dx_layer = convert_padded_derivs_layer(input_layer, dx_layer, pool_size, stride_size, padding_fn)
        return dx_layer
    elif dx == None:
        return pooled_input_layer



# # def conv_layer_backprop(kernel, delta, prev_layer, strides):
# #     kernel_rows = kernel.shape[0]
# #     kernel_cols = kernel.shape[1]
# #     num_rows = delta.shape[0]
# #     num_cols = delta.shape[1]
# #     in_channels = delta.shape[2]
    
# #     kernel_deriv = np.zeros(kernel.shape)
# #     for row in range(num_rows):
# #         for cols in range(num_cols):
# #             cur_row = (strides[0] * row)
# #             cur_col = (strides[1] * col)
# #             kernel_deriv += (delta[row,col] * prev_layer[cur_row: (kernel_rows + cur_row), cur_col: (kernel_cols + cur_col)])







# # NNconv = neuralnetwork.ConvNeuralNetwork(
# #     layers=["conv, pool, conv, pool"],
# #     num_filters=[32, 64],
# #     kernel_sizes=[[5, 5, 3], [5, 5, 32]],
# #     stride_sizes=[[2, 2], [2, 2], [2, 2], [2, 2]],
# #     pool_sizes=[[2, 2], [2, 2]],
# #     pool_methods=["max", "mean"],
# #     activations=["relu", "relu"],
# #     paddings=["same", "same"]
# #     )
# # NN = neuralnetwork.NeuralNetwork(sizes=[1, 512, 1], activations=["linear", "relu", "linear"], scale_method="normalize", optimizer="nadam", lr=0.001)

# inputs = np.random.rand(1, 9, 9, 1)
# kernels = np.random.rand(1, 2, 2, 1)
# biases = np.random.rand(1, 1, 1, 1)
# conv = generate_fmaps(inputs, kernels, biases, [2, 2], "valid", "relu", dx=None)
# pool = pool_fmaps(conv, [2,2], [2,2], "max", dx="input_layer")
# print(conv.shape)
# print(pool.shape)
# print(pool_fmaps(conv, [2,2], [2,2], "max"))

layer_names = ["conv", "pool", "conv", "pool"]
in_channels = [3, None, 32, None]
num_filters=[32, None, 4, None]
kernel_sizes=[[2, 2], None, [2, 2], None]
stride_sizes=[[2, 2], [2, 2], [2, 2], [2, 2]]
pool_sizes=[None, [2, 2], None, [2, 2]]
pool_fns=[None, "max", None, "max"]
activation_fns=["relu", None, "relu", None]
padding_fns = ["same", "valid", "same", "valid"]

kernel_layers = []
bias_layers = []

for layer_idx in range(len(layer_names)):
    if (layer_names[layer_idx] == "conv"):
        kernel_rows, kernel_cols, num_in_channels = kernel_sizes[layer_idx][0], kernel_sizes[layer_idx][1], in_channels[layer_idx]
        layer_filters_num = num_filters[layer_idx]
        kernel_layers.append(np.random.rand(layer_filters_num, kernel_rows, kernel_cols, num_in_channels))
        bias_layers.append(np.random.rand(layer_filters_num, 1, 1, 1))
    else:
        kernel_layers.append(None)
        bias_layers.append(None)


def conv_feedforward(input_layer, return_aLayers=True):
    a = input_layer
    for layer_name, kernel_layer, bias_layer, stride_size, pool_size, pool_fn, activation_fn, padding_fn in zip(layer_names, kernel_layers, bias_layers, stride_sizes, pool_sizes, pool_fns, activation_fns, padding_fns):
        if layer_name == "conv":
            a = generate_fmaps(a, kernel_layer, bias_layer, stride_size, padding_fn, activation_fn, dx=None)
        elif layer_name == "pool":
            a = pool_fmaps(a, pool_size, stride_size, pool_fn, padding_fn, dx=None)
    return a

def conv_backprop(input_layer, dx="kernel_layer", dx_layer_num=1):
    pools = []
    kernels = []
    aLayers = [input_layer]
    a = input_layer
    for layer_name, kernel_layer, bias_layer, stride_size, pool_size, pool_fn, activation_fn, padding_fn in zip(layer_names, kernel_layers, bias_layers, stride_sizes, pool_sizes, pool_fns, activation_fns, padding_fns):
        if layer_name == "conv":
            if len(aLayers) == dx_layer_num:
                kernel = generate_fmaps(a, kernel_layer, bias_layer, stride_size, padding_fn, activation_fn, dx=dx)
            else:
                kernel = generate_fmaps(a, kernel_layer, bias_layer, stride_size, padding_fn, activation_fn, dx="input_layer")
            a = generate_fmaps(a, kernel_layer, bias_layer, stride_size, padding_fn, activation_fn, dx=None)
            kernels.append(kernel)
        elif layer_name == "pool":
            pool = pool_fmaps(a, pool_size, stride_size, pool_fn, padding_fn, dx="input_layer")
            a = pool_fmaps(a, pool_size, stride_size, pool_fn, padding_fn, dx=None)
            kernels.append(pool)
        aLayers.append(a)
        
    deltas = np.random.rand(1, 6, 6, 4)
    #a = generate_fmaps(deltas, kernel_layers[-2], bias_layers[-2], stride_sizes[-2], padding_fns[-2], activation_fns[-2], dx=None)
    for idx in range(1, len(kernels) + 1):
        print(kernels[-idx].shape)
        # if (layer_names[-idx] == "pool"):
        #     np.place(kernels[-idx], kernels[-idx] == 1, deltas)
        #     deltas = kernels[-idx]
        # else:
        #     deltas = np.sum(deltas, 3)*np.sum(kernels[-idx],3)
        # x = np.copy(pools[pool][:,:,0])
        # after = np.place(pools[pool][:,:,0], pools[pool][:,:,0] == 1, np.random.rand)
        # print(x, pools[pool][:,:,0])
    print(deltas.shape)
    return a

input_layer = np.random.rand(1, 24, 24, 3)
forward = conv_feedforward(input_layer)
backward = conv_backprop(input_layer, dx="kernel_layer", dx_layer_num=1)
#print(forward)
# def conv_backprop(inputs, delta):
#     kernel_derivs = np.zeros((kernel_layers.shape))
#     bias_derivs = np.zeros((bias_layers.shape))

#     aLayers = conv_feedforward(inputs, return_aLayers=True)

#     for l in range(1, len(layer_names)):
#         if layer_names[-l] == "pool":
#             delta *= pool_fmaps(aLayers[-l-1], return_input_derivs=True)



# inputs = np.random.rand(14, 28, 28, 3)

# kernel_layers = []
# biase_layers = []
# for num_filters, kernel_size, padding_method in zip(num_filters, kernel_sizes, paddings):
#     kernel_size = kernel_size
#     if padding_method == "same":
#         bias_size

# for layer in num_filters:
    
# kernels = [[np.random.rand(kernel_size[0], kernel_size[1], kernel_size[2]) for _ in range(layer) for kernel_size in kernel_sizes] for layer in num_filters]
# biases = [[np.random.rand() for _ in range(layer)] for layer in num_filters] 
# kernelsLayer1 = np.array([np.random.rand(5, 5, 3) for _ in range(32)])
# fmaps = generate_fmaps(inputs, kernelsLayer1, [1,1], "same")
# print(fmaps.shape)
# pooled_fmaps = pool_fmaps(fmaps, [2, 2], [2, 2])
# print(pooled_fmaps.shape)

