import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential

# Define the 5x5 input matrix
input_matrix = np.array([[1, 2, 3, 4, 5],
                         [6, 7, 8, 9, 10],
                         [11, 12, 13, 14, 15],
                         [16, 17, 18, 19, 20],
                         [21, 22, 23, 24, 25]], dtype=np.float32)

# Define the custom 3x3 Laplacian kernel
kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]], dtype=np.float32)

# Function to perform convolution with different parameters
def convolution(input_matrix, kernel, stride, padding):
    # Reshape the input matrix to add a batch dimension and a channel dimension
    input_matrix_reshaped = input_matrix.reshape(1, 5, 5, 1)

    # Build a simple model with a Conv2D layer
    model = Sequential([
        Conv2D(filters=1, kernel_size=(3, 3), strides=stride, padding=padding, use_bias=False, input_shape=(5, 5, 1))
    ])

    # Set the kernel weights
    model.layers[0].set_weights([kernel.reshape(3, 3, 1, 1)])

    # Perform the convolution
    output = model.predict(input_matrix_reshaped)

    return output.squeeze()  # Remove the extra dimensions

# Stride = 1, Padding = 'VALID'
output_valid_stride_1 = convolution(input_matrix, kernel, stride=(1, 1), padding='valid')
print("Output with Stride = 1, Padding = 'VALID':\n", output_valid_stride_1)

# Stride = 1, Padding = 'SAME'
output_same_stride_1 = convolution(input_matrix, kernel, stride=(1, 1), padding='same')
print("\nOutput with Stride = 1, Padding = 'SAME':\n", output_same_stride_1)

# Stride = 2, Padding = 'VALID'
output_valid_stride_2 = convolution(input_matrix, kernel, stride=(2, 2), padding='valid')
print("\nOutput with Stride = 2, Padding = 'VALID':\n", output_valid_stride_2)

# Stride = 2, Padding = 'SAME'
output_same_stride_2 = convolution(input_matrix, kernel, stride=(2, 2), padding='same')
print("\nOutput with Stride = 2, Padding = 'SAME':\n", output_same_stride_2)
