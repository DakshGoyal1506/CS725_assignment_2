import numpy as np

def Convolution_1D(
    array1: np.array, array2: np.array, padding: str, stride: int = 1
) -> np.array:
    """
    Compute the convolution array1 * array2.
    
    Args:
    array1: np.array, input array
    array2: np.array, kernel
    padding: str, padding type ('full' or 'valid')
    stride: int, specifies how much we move the kernel at each step
    
    Carefully look at the formula for convolution in the problem statement, specifically the g[n-m] term.
    What does it indicate?
    Also note the constraints on sizes:
    - For padding='full', the sizes of array1 and array2 can be anything
    - For padding='valid', the size of array1 must be greater than or equal to the size of array2.
    
    Returns:
    np.array, output of the convolution
    """
    # revering filter
    array2 = array2[::-1]
    # Apply padding if needed
    if padding == 'full':
        pad_size = len(array2) - 1
        array1_padded = np.pad(array1, (pad_size, pad_size), mode='constant')
    else:  # 'valid' padding
        array1_padded = array1

    # Calculate output length
    if padding == 'full':
        output_length = (len(array1) + len(array2) - 1 - 1) // stride + 1
    else:  # 'valid' padding
        output_length = (len(array1) - len(array2)) // stride + 1

    # Initialize the output array
    output = np.zeros(output_length)

    # Perform the convolution
    for i in range(output_length):
        start = i * stride
        end = start + len(array2)
        output[i] = np.sum(array1_padded[start:end] * array2)
    return output


def probability_sum_of_faces(p_A: np.array, p_B: np.array) -> np.array:
    """
    Compute the probability of the sum of faces of two biased dice rolled together.
    
    Args:
        p_A (np.array): Probabilities of the faces of die A.
        p_B (np.array): Probabilities of the faces of die B.
    
    Returns:
        np.array: Probabilities of the sum of faces of die A and die B.
    """
    # Perform convolution using the Convolution_1D function
    # The probability mass function of the sums starts from 2 (index 0) to n+m (index n+m-2)
    result = Convolution_1D(p_A, p_B, padding='full')
    
    return result


if __name__ == "__main__":

#(A) 1D Conv.
    array1 = np.array([1, 2, 3])
    array2 = np.array([0, 1, 0.5])
    # print("Full:", array2)  # Expected output for full
    output_full = Convolution_1D(array1, array2, padding='full')
    output_valid = Convolution_1D(array1, array2, padding='valid')
    print("Full:", output_full)  # Expected output for full
    print("Valid:", output_valid)  # Expected output for valid
    p_A = np.array([0.1, 0.2, 0.4, 0.1, 0.1, 0.2])  # 6-faced die A
    p_B = np.array([0.3, 0.4, 0.3])                  # 3-faced die B
    file_name = "Conv_1D.txt"
    with open(file_name, "w") as file:
        file.write(f"Full: {output_full}\n")  # Write full output
        file.write(f"Valid: {output_valid}\n")  # Write valid output

#(B) Two dice.
    probabilities = probability_sum_of_faces(p_A, p_B)
    print(probabilities)  # Expected output: [0.03, 0.1, 0.2, 0.21, 0.16, 0.13, 0.11, 0.06]
    
    file_name = "Two_Dice.txt"
    with open(file_name, "w") as file:
        file.write(f"Full: {probabilities}\n")  # Write full output
    # print(sum(probabilities))
