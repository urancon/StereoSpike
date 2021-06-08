import numpy as np


def CONV_synaptic_ops(in_size: tuple, channels_out: int, kernel_size):
    """
    Returns the number of synaptic operations (SOP) in a spiking convolutional layer.
    """
    Cin, Hin, Win = in_size
    Cout = channels_out
    K = kernel_size

    # 1. Number of synaptic operations in a convolutional layer with Cin = Cout = 1
    SOP_map = np.zeros((Hin, Win))

    # 1.a) calculate top left quadrant
    if (Win % 2) == 0:
        for i in range(Win // 2):
            SOP_map[0][i] = min(i + 1, K)  # top row
    else:
        for i in range(Win // 2 + 1):
            SOP_map[0][i] = min(i + 1, K)

    if (Hin % 2) == 0:
        for j in range(Hin // 2):
            SOP_map[j] = SOP_map[0] * min((j + 1), K)  # following rows are multiples of the first
    else:
        for j in range(Hin // 2 + 1):
            SOP_map[j] = SOP_map[0] * min((j + 1), K)

    # 1.b) top right quadrant = symmetry of 1.a) along vertical axis
    SOP_map[0:Hin // 2 + 1, Win // 2 + 1:Win] = np.flip(SOP_map[0:Hin // 2 + 1, 0:Win // 2], axis=1)

    # 1.c) bottom = horizontal symmetry of 1.b) along horizontal axis
    SOP_map[Hin // 2 + 1:Hin, :] = np.flip(SOP_map[0:Hin // 2, :], axis=0)

    # 1.d) calculation of the number of synaptic operations
    SOP_number = np.sum(SOP_map)

    # 2. Number of synaptic operations in a convolutional layer with Cin >= 1 and Cout = 1
    SOP_number *= Cin

    # 3. Number of synaptic operations in a convolutional layer with Cin >= 1 and Cout >= 1
    SOP_number *= Cout

    return SOP_number


def N2O_synaptic_ops(in_size, channels_out):
    """
    Returns the number of synaptic operations in an N-to-One layer
    """
    SOP_number = channels_out
    for dim in in_size:
        SOP_number *= dim
    return SOP_number


if __name__ == "__main__":
    n_conv = CONV_synaptic_ops(in_size=(1, 9, 9), channels_out=1, kernel_size=1)
    print(n_conv)
    n_n2o = N2O_synaptic_ops((32, ))
    print(n_n2o)
