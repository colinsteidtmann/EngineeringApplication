import numpy as np 
class ConvNet:
    def __init__(self, layer_names, input_channels, num_filters, kernel_sizes, stride_sizes, pool_sizes, pool_fns, activations, padding_fns):
        self.layer_names = layer_names
        self.kernel_channels = [input_channels] + [last_num_filters for last_num_filters in num_filters[1:]]
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.stride_sizes = stride_sizes
        self.pool_sizes = pool_sizes
        self.pool_fns = pool_fns
        self.activations = activation_fns
        self.padding_fns = padding_fns
        self.init_paramaters()
    
    def init_paramaters(self):
        self.kernel_layers = []
        self.bias_layers = []

        for layer_idx in range(len(self.layer_names)):
            if (self.layer_names[layer_idx] == "conv"):
                kernel_rows, kernel_cols, kernel_channels = self.kernel_sizes[layer_idx][0], self.kernel_sizes[layer_idx][1], self.kernel_channels[layer_idx]
                layer_filters_num = self.num_filters[layer_idx]
                self.kernel_layers.append(np.random.rand(layer_filters_num, kernel_rows, kernel_cols, kernel_channels))
                self.bias_layers.append(np.random.rand(layer_filters_num, 1, 1, 1))
            else:
                self.kernel_layers.append(None)
                self.bias_layers.append(None)

    def conv_feedforward(self, input_layer):
        a = input_layer
        for idx, layer_name in enumerate(self.layer_names):
            if layer_name == "conv":
                a = pad(a, idx)
                a = convolution(a, idx)
            elif layer_name == "pool":
                a = pad(a, idx)
                a = pool(a, idx)
        return a

    def gradients(self, input_layer, dx_layer, dx_type, grad_ys=None):
        """ Calculates derivitaves of dx_layer w.r.t output
            x = input, optional y = label
            ``dx_layer`` & ``dy_layer`` are integers (or list of intergers for dx_layer), layer numbers starting at 1. Weights and biases 
                                        have the same layer number as the hidden or output layer ahead of them.
            ``dx_type`` is the type of layer for dx_layer, it can be "weights", "biases", "input", "hidden" or "output"
            ``dy_type`` is the type of layer for dy_layer, it can only be "output" or "loss" 
            ``grad_ys`` represent the "starting" backprop value, ones if set to None, must be the same shape as dy
            returns a list of gradients or single gradient (depending on if dx_layer is a list or not)
        """

        """ arrays to store outputs of each layer """
        dropoutLayers = []
        zLayers = [x]  
        aLayers = [x]
        
        """ feedforward with input x, store outputs, z, and activations of z of each layer 
            (if using dropout regularzation store the the array with 0's (dropout mask), else just the default dropout array with 1's 
            so that no neurons are dropped)
        """
        for idx, layer_name in enumerate(self.layer_names):
            if layer_name == "conv":
                a = pad(a_layers[-1], idx)
                a_layers.append(a)
                a = convolution(a_layers[-1], idx)
                a_layers.append(a)
            elif layer_name == "pool":
                a = pad(a_layers[-1], idx)
                a_layers.append(a)
                a = pool(a_layers[-1], idx)
                a_layers.append(a)

        """ Begin Backpropagation
            Multiply grad_ys, "starting" backprop value, times delta
            get d of cost w.r.t final layer,  δᴸ = ∇ₐC ⊙ σ′(zᴸ) or ....
            get d of final layer w.r.t final layer,  δᴸ = σ′(zᴸ) 
            Multiply δᴸ * the output scale derivative            
        """
        grad_ys = np.ones(np.array(aLayers[-1]).shape) if (grad_ys == None) else np.array(grad_ys).reshape(np.array(a_layers[-1]).shape)

        """ backwards """
        delta_layers = [delta_layer]
        kernel_derivs = []
        bias_derivs = []
        for layer_idx, layer_name in enumerate(reversed(self.layer_names)):
            if layer_name == "conv":
                idx = len(delta_layers)
                kernel_deriv = get_kernel_derivs(a_layers[-idx - 1], delta_layers[-1], layer_idx)
                bias_deriv = np.sum(delta_layers[-1], axis=(0,1,2)).reshape(bias_layer.shape)
                delta = get_delta_derivs(a_layers[-idx - 1], kernel_layer, delta_layers[-1], layer_idx)
                delta_layers.append(delta)
                bias_derivs.insert(0,bias_deriv)
                kernel_derivs.insert(0,kernel_deriv)
                
                idx = len(delta_layers)
                delta = get_pad_derivs(a_layers[-idx-1], delta_layers[-1], layer_idx)
                delta_layers.append(delta)
            elif layer_name == "pool":
                idx = len(delta_layers)
                delta = get_pool_derivs(a_layers[-idx-1], a_layers[-idx], delta_layers[-1], layer_idx)
                delta_layers.append(delta)
                idx = len(delta_layers)
                delta = get_pad_derivs(a_layers[-idx-1], delta_layers[-1], layer_idx)
                delta_layers.append(delta)

        def convolution(self, input_layer, layer_idx):
            """
                input_layer == whatever input
                layer_idx == input_layer idx
                returns convolved input_layer
            """

            """ Set local variables """
            kernel_layer = self.kernel_layers[layer_idx]
            bias_layer = self.bias_layers[layer_idx]
            stride_size = self.stride_sizes[layer_idx]

            """ Start convolving input layer """
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

        def pool(self, input_layer, layer_idx):
            """
                input_layer == whatever input
                layer_idx == input_layer idx
                returns pooled input_layer
            """

            """ Set local variables """
            pool_size = self.pool_sizes[layer_idx]
            stride_size = self.stride_sizes[layer_idx]
            pool_fn = self.pool_fns[layer_idx]

            """ Start pooling input layer """
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

        def pad(input_layer, layer_idx):
            """
                input_layer == whatever input
                layer_idx == input_layer idx
                returns padded input_layer
            """

            """ Set local variables """
            if (self.layer_names[layer_idx] == "conv"):
                filter_size = [self.kernel_layers[layer_idx].shape[1], self.kernel_layers[layer_idx].shape[2]]
            elif (self.layer_names[layer_idx] == "pool"):
                filter_size = self.pool_sizes[layer_idx]
            stride_size = self.stride_sizes[layer_idx]
            padding_fn = self.padding_fns[layer_idx]
            
            """ Start padding input layer """
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


        def get_delta_derivs(input_layer, delta_layer, layer_idx):
            """
                input_layer == zˡ
                delta_layer == δˡ⁺¹
                layer_idx == input_layer idx
                returns dδˡ⁺¹ w.r.t zˡ
            """

            """ Set local variables """
            kernel_layer = self.kernel_layers[layer_idx]
            stride_size = self.stride_sizes[layer_idx]

            """ Start getting dδˡ⁺¹ w.r.t input_layer --> producing  δˡ """
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

        def get_kernel_derivs(input_layer, delta_layer, layer_idx):
            """
                input_layer == zˡ⁻¹
                delta_layer == δ
                layer_idx == kernel_layer idx
                returns dδˡ⁺¹ w.r.t kernelsˡ

            """

            """ Set local variables """
            kernel_shape = self.kernel_layers[layer_idx].shape
            stride_size = self.stride_sizes[layer_idx]

            """ Start getting dδˡ⁺¹ w.r.t kernelsˡ """
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

        def get_pool_derivs(input_layer, pool_layer, delta_layer, layer_idx):
            """ input_layer == pre-pool
                pool_layer == post-pool
                delta_layer == dδˡ⁺¹ w.r.t pool_layer
                (delta_layer.shape == pool_layer.shape)
                layer_idx = pool_layer idx

                returns dδˡ⁺¹ w.r.t input_layerˡ
            """
            
            """ Set local variables """
            pool_size = self.pool_sizes[layer_idx]
            stride_size = self.stride_sizes[layer_idx]
            pool_fn = self.pool_fns[layer_idx]

            """ Start getting dδˡ⁺¹ w.r.t input_layerˡ """
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

        def get_pad_derivs(orig_input_layer, delta_layer, layer_idx):
            """ orig_input_layer == pre-pad
                delta_layer == dδˡ⁺¹ w.r.t post_pad_layer-pad_layer
                (delta_layer.shape == post_pad_layer.shape)
                layer_idx = pad_layer idx

                returns dδˡ⁺¹ w.r.t orig_input_layerˡ
            """

            """ Set local variables """
            if (self.layer_names[layer_idx] == "conv"):
                filter_size = [self.kernel_layers[layer_idx].shape[1], self.kernel_layers[layer_idx].shape[2]]
            elif (self.layer_names[layer_idx] == "pool"):
                filter_size = self.pool_sizes[layer_idx]
            stride_size = self.stride_sizes[layer_idx]
            padding_fn = self.padding_fns[layer_idx]

            """ Start getting dδˡ⁺¹ w.r.t orig_input_layerˡ """
            num_inputs, input_rows, input_cols, num_channels = orig_input_layer.shape[0], orig_input_layer.shape[1], orig_input_layer.shape[2], orig_input_layer.shape[3]
            filter_rows, filter_cols, stride_rows, stride_cols = filter_size[0], filter_size[1], stride_size[0], stride_size[1]
            if (padding_fn == "same"):
                width_padding = (input_cols * stride_cols) + filter_cols - input_cols - stride_cols
                height_padding = (input_rows * stride_rows) + filter_rows - input_rows - stride_rows
                row_padding_right = int(np.ceil(width_padding / 2))
                row_padding_left = int(np.floor(width_padding / 2))
                col_padding_bottom = int(np.ceil(height_padding / 2))
                col_padding_top = int(np.floor(height_padding / 2))
                padded_inputs_derivs = delta_layer[:, col_padding_top:-col_padding_bottom, row_padding_left:-row_padding_right, :]
            elif (padding_fn == "valid"):
                max_num_rows = (int)((input_rows - filter_rows) / stride_rows) + 1
                max_num_cols = (int)((input_cols - filter_cols) / stride_cols) + 1
                cut_bottom_rows = input_rows - (filter_rows + (stride_rows * (max_num_rows - 1)))
                cut_right_cols = input_cols - (filter_cols + (stride_cols * (max_num_cols - 1)))
                padded_inputs_derivs = np.pad(delta_layer, [(0,0), (0, cut_bottom_rows), (0, cut_right_cols), (0, 0)], mode='constant')
            return padded_inputs_derivs