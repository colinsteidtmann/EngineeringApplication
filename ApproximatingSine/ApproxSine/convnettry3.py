import numpy as np 
""" FeedForward """
def activation_fn(number):
    if (type(number) == np.ndarray):
        number2 = np.copy(number)
        number2[number2 < 0] = 0
        return number2
    elif number < 0:
        return 0
    else:
        return number

def activation_fn_deriv(number):
    if number <= 0:
        return 0
    else:
        return 1

def convolution(input_layer, kernel_layer, bias_layer, stride_size):
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
    num_inputs, num_rows, num_cols = input_layer.shape[0], input_layer.shape[1], input_layer.shape[2]
    num_filters, kernel_rows, kernel_cols, num_channels = kernel_layer.shape[0], kernel_layer.shape[1], kernel_layer.shape[2], kernel_layer.shape[3]
    stride_rows, stride_cols = stride_size[0], stride_size[1]
    output_rows, output_cols = int(((num_rows - kernel_rows) / stride_rows) + 1), int(((num_cols - kernel_cols) / stride_cols) + 1)
    output_layer = np.zeros((num_inputs, output_rows, output_cols, num_filters))
    for inputs_idx in range(num_inputs):
        for kernel_idx in range(num_filters):
            for output_row in range(output_rows):
                for output_col in range(output_cols):
                    z = np.squeeze((activation_fn(input_layer[inputs_idx, stride_rows*output_row:stride_rows*output_row+kernel_rows, stride_cols*output_col:stride_cols*output_col+kernel_cols, :]) * np.rot90(kernel_layer[kernel_idx, :, :, :],2)).sum())
                    z += bias_layer[kernel_idx]
                    output_layer[inputs_idx, output_row, output_col, kernel_idx] = z 
    return output_layer

def pool(input_layer, pool_size, stride_size, pool_fn):
    num_inputs, num_rows, num_cols, num_channels = input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], input_layer.shape[3]
    pool_rows, pool_cols, stride_rows, stride_cols = pool_size[0], pool_size[1], stride_size[0], stride_size[1]
    output_pool_rows = int(((num_rows - pool_rows) / stride_rows) + 1)
    output_pool_cols = int(((num_cols - pool_cols) / stride_cols) + 1)
    output_layer = np.zeros((num_inputs, output_pool_rows, output_pool_cols, num_channels))
    for inputs_idx in range(num_inputs):
        for channel_idx in range(num_channels):
            for output_pool_row_idx in range(output_pool_rows):
                for output_pool_col_idx in range(output_pool_cols):
                    input_start_row, input_start_col = (stride_rows * output_pool_row_idx), (stride_cols * output_pool_col_idx)
                    z = np.squeeze(np.max(input_layer[inputs_idx, input_start_row:(input_start_row + pool_rows), input_start_col:(input_start_col + pool_cols), channel_idx]))
                    output_layer[inputs_idx, output_pool_row_idx, output_pool_col_idx, channel_idx] = z
    return output_layer

def pad(input_layer, filter_size, stride_size, padding_fn):
    num_inputs, input_rows, input_cols, num_channels = input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], input_layer.shape[3]
    filter_rows, filter_cols, stride_rows, stride_cols = filter_size[0], filter_size[1], stride_size[0], stride_size[1]
    if (padding_fn == "same"):
        width_padding = (input_cols * stride_cols) + filter_cols - input_cols - stride_cols
        height_padding = (input_rows * stride_rows) + filter_rows - input_rows - stride_rows
        row_padding_right = int(np.ceil(width_padding / 2))
        row_padding_left = int(np.floor(width_padding / 2))
        col_padding_bottom = int(np.ceil(height_padding / 2))
        col_padding_top = int(np.floor(height_padding / 2))
        padded_inputs = np.pad(input_layer, [(0,0), (col_padding_top, col_padding_bottom), (row_padding_left, row_padding_right), (0,0)], mode='constant')
    elif (padding_fn == "valid"):
        max_num_rows = (int)((input_rows - filter_rows) / stride_rows) + 1
        max_num_cols = (int)((input_cols - filter_cols) / stride_cols) + 1
        padded_inputs = input_layer[:, :(filter_rows + (stride_rows * (max_num_rows - 1))), :(kernel_cols + (stride_cols * (max_num_cols - 1))), :]
    return padded_inputs

""" Backpropagation """
def get_delta_derivs(input_layer, kernel_layer, delta_layer, stride_size):
    num_inputs, input_rows, input_cols, input_channels = input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], input_layer.shape[3]
    num_filters, kernel_rows, kernel_cols = kernel_layer.shape[0], kernel_layer.shape[1], kernel_layer.shape[2]
    delta_rows, delta_cols = delta_layer.shape[1], delta_layer.shape[2]
    stride_rows, stride_cols = stride_size[0], stride_size[1] 
    delta_derivs = np.zeros(input_layer.shape)
    for inputs_idx in range(num_inputs):
        for inputs_channel in range(input_channels):
            for kernel_idx in range(num_filters):
                for delta_row in range(delta_rows):
                    for delta_col in range(delta_cols):
                        for kernel_row in range(kernel_rows):
                            for kernel_col in range(kernel_cols):
                                z = kernel_layer[kernel_idx, -kernel_row - 1, -kernel_col - 1, inputs_channel] * delta_layer[inputs_idx, delta_row, delta_col, kernel_idx]
                                z *= activation_fn_deriv(input_layer[inputs_idx, stride_rows * delta_row + kernel_row, stride_cols * delta_col + kernel_col, inputs_channel])
                                delta_derivs[inputs_idx, stride_rows * delta_row + kernel_row, stride_cols * delta_col + kernel_col, inputs_channel] += z
    return delta_derivs

def get_kernel_derivs(input_layer, delta_layer, stride_size, kernel_shape):
    num_filters, kernel_rows, kernel_cols, kernel_channels = kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3]
    num_inputs, delta_rows, delta_cols = delta_layer.shape[0], delta_layer.shape[1], delta_layer.shape[2]
    stride_rows, stride_cols = stride_size[0], stride_size[1]
    kernel_derivs = np.zeros(kernel_shape)   
    for kernel_idx in range(num_filters):
        for kernel_row in range(kernel_rows):
            for kernel_col in range(kernel_cols):
                for kernel_channel in range(kernel_channels):
                    kernel_gradient_sum = 0
                    for inputs_idx in range(num_inputs):
                        for delta_row in range(delta_rows):
                            for delta_col in range(delta_cols):
                                z = activation_fn(input_layer[inputs_idx, stride_rows*delta_row+kernel_row, stride_cols*delta_col+kernel_col, kernel_channel]) * delta_layer[inputs_idx, delta_row, delta_col, kernel_idx]
                                kernel_gradient_sum += z
                    kernel_derivs[kernel_idx, -kernel_row-1, -kernel_col-1, kernel_channel] = kernel_gradient_sum
    return kernel_derivs

def get_pool_derivs(input_layer, pool_layer, delta_layer, pool_size, stride_size, pool_fn):
    pool_rows, pool_cols, stride_rows, stride_cols = pool_size[0], pool_size[1], stride_size[0], stride_size[1]
    num_inputs, num_pool_rows, num_pool_cols, num_channels = pool_layer.shape[0], pool_layer.shape[1], pool_layer.shape[2], pool_layer.shape[3]
    input_layer_derivs = np.zeros(input_layer.shape)
    for inputs_idx in range(num_inputs):
        for channel_idx in range(num_channels):
            for pool_row_idx in range(num_pool_rows):
                for pool_col_idx in range(num_pool_cols):
                    delta_val = delta_layer[inputs_idx, pool_row_idx, pool_col_idx, channel_idx]
                    pool_val = pool_layer[inputs_idx, pool_row_idx, pool_col_idx, channel_idx]
                    input_start_row, input_start_col = (stride_rows * pool_row_idx), (stride_cols * pool_col_idx)
                    r, c, = np.squeeze(np.where(np.isclose(input_layer[inputs_idx, input_start_row:(input_start_row + pool_rows), input_start_col:(input_start_col + pool_cols), channel_idx], pool_val)))
                    input_layer_derivs[inputs_idx, input_start_row + r, input_start_col + c, channel_idx] = delta_val
    return input_layer_derivs

def get_pad_derivs(orig_input_layer, pad_input_layer, filter_size, stride_size, padding_fn):
    num_inputs, input_rows, input_cols, num_channels = orig_input_layer.shape[0], orig_input_layer.shape[1], orig_input_layer.shape[2], orig_input_layer.shape[3]
    filter_rows, filter_cols, stride_rows, stride_cols = filter_size[0], filter_size[1], stride_size[0], stride_size[1]
    if (padding_fn == "same"):
        width_padding = (input_cols * stride_cols) + filter_cols - input_cols - stride_cols
        height_padding = (input_rows * stride_rows) + filter_rows - input_rows - stride_rows
        row_padding_right = int(np.ceil(width_padding / 2))
        row_padding_left = int(np.floor(width_padding / 2))
        col_padding_bottom = int(np.ceil(height_padding / 2))
        col_padding_top = int(np.floor(height_padding / 2))
        padded_inputs_derivs = pad_input_layer[:, col_padding_top:-col_padding_bottom, row_padding_left:-row_padding_right, :]
    elif (padding_fn == "valid"):
        max_num_rows = (int)((input_rows - filter_rows) / stride_rows) + 1
        max_num_cols = (int)((input_cols - filter_cols) / stride_cols) + 1
        cut_bottom_rows = input_rows - (filter_rows + (stride_rows * (max_num_rows - 1)))
        cut_right_cols = input_cols - (filter_cols + (stride_cols * (max_num_cols - 1)))
        padded_inputs_derivs = np.pad(pad_input_layer, [(0,0), (0, cut_bottom_rows), (0, cut_right_cols), (0, 0)], mode='constant')
    return padded_inputs_derivs




layer_names = ["conv", "pool", "conv", "pool"]
in_channels = [3, None, 3, None]
num_filters=[3, None, 2, None]
kernel_sizes=[[5, 5], None, [5, 5], None]
stride_sizes=[[1, 1], [2, 2], [1, 1], [2, 2]]
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

def conv_feedforward(input_layer):
    a = input_layer
    for layer_name, kernel_layer, bias_layer, stride_size, pool_size, pool_fn, activation_fn_str, padding_fn in zip(layer_names, kernel_layers, bias_layers, stride_sizes, pool_sizes, pool_fns, activation_fns, padding_fns):
        if layer_name == "conv":
            a = pad(a, [kernel_layer.shape[1], kernel_layer.shape[2]], stride_size, padding_fn)
            a = convolution(a, kernel_layer, bias_layer, stride_size)
        elif layer_name == "pool":
            a = pad(a, pool_size, stride_size, padding_fn)
            a = pool(a, pool_size, stride_size, pool_fn)
    return a

def conv_backprop(input_layer, delta_layer):
    """ forwards """
    a_layers = [input_layer]
    for layer_name, kernel_layer, bias_layer, stride_size, pool_size, pool_fn, activation_fn_str, padding_fn in zip(layer_names, kernel_layers, bias_layers, stride_sizes, pool_sizes, pool_fns, activation_fns, padding_fns):
        if layer_name == "conv":
            a = pad(a_layers[-1], [kernel_layer.shape[1], kernel_layer.shape[2]], stride_size, padding_fn)
            a_layers.append(a)
            a = convolution(a_layers[-1], kernel_layer, bias_layer, stride_size)
            print(a.shape)
            a_layers.append(a)
        elif layer_name == "pool":
            a = pad(a_layers[-1], pool_size, stride_size, padding_fn)
            a_layers.append(a)
            a = pool(a_layers[-1], pool_size, stride_size, pool_fn)
            a_layers.append(a)
    
    """ backwards """
    delta_layers = [delta_layer]
    kernel_derivs = []
    bias_derivs = []
    for layer_name, kernel_layer, bias_layer, stride_size, pool_size, pool_fn, activation_fn_str, padding_fn in zip(reversed(layer_names), reversed(kernel_layers), reversed(bias_layers), reversed(stride_sizes), reversed(pool_sizes), reversed(pool_fns), reversed(activation_fns), reversed(padding_fns)):
        if layer_name == "conv":
            idx = len(delta_layers)
            kernel_deriv = get_kernel_derivs(a_layers[-idx - 1], delta_layers[-1], stride_size, kernel_layer.shape)
            bias_deriv = np.sum(delta_layers[-1], axis=(0,1,2)).reshape(bias_layer.shape)
            delta = get_delta_derivs(a_layers[-idx - 1], kernel_layer, delta_layers[-1], stride_size)
            delta_layers.append(delta)
            bias_derivs.insert(0,bias_deriv)
            kernel_derivs.insert(0,kernel_deriv)
            
            idx = len(delta_layers)
            delta = get_pad_derivs(a_layers[-idx-1], delta_layers[-1], [kernel_layer.shape[1], kernel_layer.shape[2]], stride_size, padding_fn)
            delta_layers.append(delta)
        elif layer_name == "pool":
            idx = len(delta_layers)
            delta = get_pool_derivs(a_layers[-idx-1], a_layers[-idx], delta_layers[-1], pool_size, stride_size, pool_fn)
            delta_layers.append(delta)
            idx = len(delta_layers)
            delta = get_pad_derivs(a_layers[-idx-1], delta_layers[-1], pool_size, stride_size, padding_fn)
            delta_layers.append(delta)

    #print(bias_derivs[0][0])
    return bias_derivs[0][0]


inputs = np.random.rand(2, 28, 28, 3)
delta = np.ones(conv_feedforward(inputs).shape)*2
my_deriv = conv_backprop(inputs, delta)
epsilon = 1e-5
bias_layers[0][0,:,:,:] += epsilon
plus_output = conv_feedforward(inputs)
bias_layers[0][0,:,:,:] -= (epsilon*2)
minus_output = conv_feedforward(inputs)
approx_deriv = (plus_output - minus_output) / (2 * epsilon)
i, r, c, ch = np.where(approx_deriv != 0)
approx_deriv = approx_deriv[i, r, c, ch].sum()
print(approx_deriv)
print(my_deriv)


# inputs = np.random.uniform(-2,1,(2, 28, 28, 3))
# kernels = np.random.rand(5, 5, 5, 3)
# biases = np.random.rand(5, 1, 1, 1)

# idx,row,col,chan = 1,5,5,2
# plus_input = np.copy(inputs)
# plus_input[idx, row, col, chan] += epsilon
# plus_input = pad(plus_input, [5,5], [1,1], "same")
# plus_output = convolution(plus_input, kernels, biases, [1, 1])
# plus_output_pool = pool(plus_output, [2,2], [2,2], "max")

# minus_input = np.copy(inputs)
# minus_input[idx, row, col, chan] -= epsilon
# minus_input = pad(minus_input, [5,5], [1,1], "same")
# minus_output = convolution(minus_input, kernels, biases, [1, 1])
# minus_output_pool = pool(minus_output, [2,2], [2,2], "max")


# deriv = (plus_output_pool - minus_output_pool)/ (2 * epsilon)
# i, r, c, ch = np.where(deriv != 0)
# deriv = deriv[i,r,c,ch].sum()

# reg_inputs_pad = pad(inputs, [5,5], [1,1], "same")
# reg_output = convolution(reg_inputs_pad, kernels, biases, [1, 1])
# reg_output_pool = pool(reg_output, [2, 2], [2, 2], "max")
# my_deriv_pool = get_pool_derivs(reg_output, reg_output_pool, np.ones(reg_output_pool.shape), [2, 2], [2, 2], "max")
# my_deriv = get_delta_derivs(reg_inputs_pad, kernels, my_deriv_pool, [1, 1])
# my_deriv_pad = get_pad_derivs(inputs, my_deriv, [5, 5], [1, 1], "same")
# my_deriv = my_deriv_pad[idx, row, col, chan]

# # print(np.abs(deriv - my_deriv))
# # if (np.abs(deriv - my_deriv) > 1e-4):
# #     print("uh oh")
# print("approx deriv = ", deriv)
# print("my_deriv = ", my_deriv)



# # input_layer = np.random.rand(1, 28, 28, 3)
# # output = conv_feedforward(input_layer)
# # delta = np.ones((1, 7, 7, 4))
# # conv_backprop(input_layer, delta)

# # epsilon=1e-5
# # plus_input = np.copy(input_layer)
# # kernel_layers[0][0,1,1,0] += epsilon
# # plus_output = conv_backprop(input_layer, delta)

# # minus_input = np.copy(input_layer)
# # kernel_layers[0][0,1,1,0] -= (epsilon*2)
# # minus_output = conv_backprop(input_layer, delta)

# # deriv = (plus_output - minus_output) / (2 * epsilon)
# # i, r, c, ch = np.where(deriv != 0)
# # deriv = deriv[i,r,c,ch].sum()
# # print("deriv = ", deriv)
# # print(output.shape)

# # inputs = [
# #     [1, 2, 3, 4, 5, 6, 7, 8, 9],
# #     [9, 8, 7, 6, 5, 4, 3, 2, 1],
# #     [6, 5, 4, 3, 2, 1, 7, 8, 9],
# #     [1, 3, 5, 7, 9, 2, 4, 6, 8]
# # ]

# # kernel = [
# #     [1, 9],
# #     [-2, -6]
# # ]

# # delta = [
# #     [2, 5, 7, 9, 3, 2, 5],
# #     [2, 4, 2, 8, 4, 6, 3]
# # ]

# # inputs = np.array(inputs).reshape((1, 4, 9, 1))
# # kernel = np.array(kernel).reshape((1, 2, 2, 1))
# # bias = np.array([3]).reshape((1,1,1,1)) 
# # z = convolution(inputs, kernel, bias, [1, 1])
# # a = activation_fn(z)
# # kernel2 = kernel
# # bias2 = bias
# # z2 = convolution(a, kernel2, bias, [1, 1])
# # a2 = activation_fn(z2)

# # delta2 = np.array(delta).reshape((1, 2, 7, 1))
# # delta1 = get_delta_derivs(z, kernel2, delta2, [1, 1])
# # delta0 = get_delta_derivs(inputs, kernel, delta1, [1,1])
# # print("a = \n", np.squeeze(a))
# # print("z = \n", np.squeeze(z))
# # print("inputs = \n", np.squeeze(inputs))
# # print("kernel = \n", np.rot90(np.squeeze(kernel), 2))
# # print("delta1 = \n", np.squeeze(delta1))
# # print("delta0 = \n", np.squeeze(delta0))


# # #feed forward
# # batch_size = 1
# # input_layerog = np.random.rand(batch_size, 28, 28, 3)
# # kernels1 = np.random.rand(32, 5, 5, 3)
# # input_layer = pad(input_layerog, kernels1, [1,1], "same")
# # biases1 = np.random.rand(32,1,1,1)
# # conv1 = convolution(input_layer, kernels1, biases1, stride_size=[1, 1])
# # pool1 = pool(conv1, pool_size=[2,2], stride_size=[2,2], pool_fn="none")

# # kernels2 = np.random.rand(64,5,5,32)
# # input_layer2 = pad(pool1, kernels2, stride_size=[1, 1], padding_fn="same")
# # biases2 = np.random.rand(64, 1, 1, 1)
# # conv2 = convolution(input_layer2, kernels2, biases2, stride_size=[1, 1])
# # pool2 = pool(conv2, pool_size=[2, 2], stride_size=[2, 2], pool_fn="none")
# # pool2_flat = pool2.reshape((batch_size, 7 * 7 * 64))


# # #backwards
# # delta = np.random.random(pool2_flat.shape)
# # delta = delta.reshape(pool2.shape)
# # pool2_deriv = get_pool_derivs(conv2, pool2, delta, [2, 2], [2, 2], "none")
# # d_wrt_padinputs2 = get_delta_derivs(kernels2, pool2_deriv, [1, 1])
# # d_wrt_originputs2 = get_pad_derivs(pool1, input_layer2, kernels2, [1, 1], "same")
# # d_wrt_kernels2 = get_kernel_derivs(input_layer2, delta, [1, 1], kernels2.shape)

# # pool1_deriv = get_pool_derivs(conv1, pool1, d_wrt_originputs2, [2, 2], [2, 2], "none")
# # d_wrt_padinputs1 = get_delta_derivs(kernels1, pool1_deriv, [1, 1])
# # d_wrt_originputs1 = get_pad_derivs(input_layerog, input_layer, kernels1, [1, 1], "same")
# # d_wrt_kernels1 = get_kernel_derivs(input_layer, d_wrt_originputs2, [1, 1], kernels1.shape)

# # print(delta.shape)
# # print(pool2_deriv.shape)
# # print(d_wrt_padinputs2.shape)
# # print(d_wrt_originputs2.shape)
# # print(d_wrt_kernels2.shape)
# # print(pool1_deriv.shape)
# # print(d_wrt_padinputs1.shape)
# # print(d_wrt_originputs1.shape)
# # print(d_wrt_kernels1.shape)