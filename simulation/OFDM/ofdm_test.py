from src.ofdm_tools import *
from src.lin_alg import *
#   Begin with frequencies
num_subcarriers = 1000
s = np.random.randint(0, 2, num_subcarriers)
# x[:10] = 1
s += (s == 0)*(-1)
channel_ir = np.random.randn(20)
# channel_ir /= np.linalg.norm(channel_ir)
channel_length = len(channel_ir)
F = DFT_matrix(num_subcarriers)
INVF = hermetian(F)
padded_channel_ir = np.concatenate((channel_ir, np.zeros(num_subcarriers-channel_length)))
channel_eigenvalues = F@padded_channel_ir
# TODO Normalize power here
x = INVF@s  # IDFT of OFDM symbol
x_cp = add_cyclic_prefix(x, channel_length)
received = np.convolve(x_cp, np.flip(channel_ir))     # Pass through LTI channel (Flips second argument)
received_fft = F@received[channel_length-1:-channel_length]
channel_equalizer = hermetian(channel_eigenvalues)  #TODO correct DFT sign?
equalized = np.real(channel_eigenvalues*received_fft)
equalized /= np.abs(equalized)
error = np.sum(equalized != s)
print(error)