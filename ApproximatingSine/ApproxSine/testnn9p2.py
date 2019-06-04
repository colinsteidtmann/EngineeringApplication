import numpy as np
import neuralnetwork_ex9 as nn

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


def pad_input_layer(input_layer, kerne, stride_size, padding_fn):
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

def activation_fn(number):
    if number < 0:
        return 0
    else:
        return number

def activation_fn_deriv(number):
    if number == 0:
        return 0
    else:
        return 1

def convolution(input_layer, kernel_layer, bias_layer, stride_size, output_shape, a_layer=None):
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
    output_layer = np.zeros(output_shape)
    num_inputs = input_layer.shape[0]
    output_rows, output_cols = output_shape[0], output_shape[1] 
    num_filters, kernel_rows, kernel_cols, num_channels = kernel_layer.shape[0], kernel_layer.shape[1], kernel_layer.shape[2], kernel_layer.shape[3]
    stride_rows, stride_cols = stride_size[0], stride_size[1]
    for inputs_idx in range(num_inputs):
        for kernel_idx in range(num_filters):
            for output_row in range(output_rows):
                for output_col in range(output_cols):
                    output_value = 0
                    for kernel_row in range(kernel_rows):
                        for kernel_col in range(kernel_cols):
                            for channel_idx in range(num_channels):                
                                z = np.squeeze(input_layer[inputs_idx, stride_rows * output_row + kernel_row, stride_cols * output_col + kernel_col, channel_idx] * kernel_layer[kernel_idx, -kernel_row - 1, -kernel_col - 1, channel_idx] + bias_layer[kernel_idx])
                                z = activation_fn(z)
                                output_value += z
                    output_layer[inputs_idx, output_row, output_col, kernel_idx] = output_value
    return output_layer

"""get_kernel_derivs works!!!!!!!!"""
def get_kernel_derivs(input_layer, delta_layer, stride_size, kernel_shape):
    num_filters, kernel_rows, kernel_cols, kernel_channels = kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3]
    num_inputs, delta_rows, delta_cols = delta_layer.shape[0], delta_layer.shape[1], delta_layer.shape[2]
    stride_rows, stride_cols = stride_size[0], stride_size[1]
    kernel_layer = np.zeros(kernel_shape)
    for kernel_idx in range(num_filters):
        for kernel_row in range(kernel_rows):
            for kernel_col in range(kernel_cols):
                for kernel_channel in range(kernel_channels):
                    kernel_gradient_sum = 0
                    for inputs_idx in range(num_inputs):
                        for delta_row in range(delta_rows):
                            for delta_col in range(delta_cols):
                                z = input_layer[inputs_idx, stride_rows*delta_row+kernel_row, stride_cols*delta_col+kernel_col, kernel_channel] * delta_layer[inputs_idx, -delta_row - 1, -delta_col - 1, kernel_idx]
                                kernel_gradient_sum += z
                    kernel_layer[kernel_idx, kernel_row, kernel_col, kernel_channel] = kernel_gradient_sum

    return kernel_layer

""" get delta derivs works too !!!!"""
def get_delta_derivs(kernel_layer, delta_layer, stride_size, input_shape):
    num_inputs, input_channels = input_shape[0], input_shape[3]
    num_filters, kernel_rows, kernel_cols = kernel_layer.shape[0], kernel_layer.shape[1], kernel_layer.shape[2]
    delta_rows, delta_cols = delta_layer.shape[1], delta_layer.shape[2]
    stride_rows, stride_cols = stride_size[0], stride_size[1]
    input_layer = np.zeros(input_shape)
    for inputs_idx in range(num_inputs):
        for inputs_channel in range(input_channels):
            for kernel_idx in range(num_filters):
                for delta_row in range(delta_rows):
                    for delta_col in range(delta_cols):
                        for kernel_row in range(kernel_rows):
                            for kernel_col in range(kernel_cols):
                                z = kernel_layer[kernel_idx, -kernel_row-1, -kernel_col-1, inputs_channel] * delta_layer[inputs_idx, delta_row, delta_col, kernel_idx]
                                input_layer[inputs_idx, stride_rows * delta_row + kernel_row, stride_cols * delta_col + kernel_col, inputs_channel] += z

    return input_layer


input_layer = np.random.rand(2, 6, 6, 2)
kernels = np.random.rand(3, 2, 2, 2)
strides = [1, 1]
biases = np.zeros((3, 1, 1, 1))
output_shape = np.zeros((2,5,5,3)).shape
output = convolution(input_layer, kernels, biases, strides, output_shape)


print("input = ", input_layer[0, :2, :2, 0])
print("kernel = ", kernels[0,:,:, 0])
print("should be ", (input_layer[0, :2, :2, 0]*np.rot90(kernels[0,:,:, 0],2)).sum() + (input_layer[0, :2, :2, 1]*np.rot90(kernels[0,:,:, 1],2)).sum())
print("output = ", output[0, 0, 0, 0])
input_layer_derivs = get_delta_derivs(kernels, output, strides, input_layer.shape)
kernel_layer_derivs = get_kernel_derivs(input_layer, output, strides, kernels.shape)
# deltas = np.random.rand(2,5,5,3)
# deriv = get_kernel_derivs(input_layer, deltas, strides, kernels.shape)
# # print("deriv = ",( input_layer[:,::2,::2,1]*deltas[:,:,:, 0]).sum())
# # print("kernel_deriv = ", deriv[0,0,0,1])
# input_deriv = get_delta_derivs(kernels, deltas, strides, input_layer.shape)
# print((kernels[:,1,1,0]*deltas[0,1,1,:]).sum() + (kernels[:,1,0,0]*deltas[0,1,2,:]).sum() + (kernels[:,0,1,0]*deltas[0,2,1,:]).sum() + (kernels[:,0,0,0]*deltas[0,2,2,:]).sum())
# print(input_deriv[0,2,2,0])

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

