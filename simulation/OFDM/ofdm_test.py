import matplotlib.pyplot as plt
from OFDM.ofdm_tools import *
#   Begin with frequencies
s = np.random.randint(0, 2, 100)
# x[:10] = 1
s += (s == 0)*(-1)
channel_ir = np.random.randn(100)
# channel_ir /= np.linalg.norm(channel_ir)
chanel_length = len(channel_ir)
F = DFT_matrix(100)
INVF = np.conjugate(F.T)
channel_fr = F@channel_ir   # Note increase in PAPR here
x = INVF@s  # Normalize power here
x_cp = add_cyclic_prefix(x, len(channel_ir))
#   Convert to sinusoids and pass through channel
received = np.convolve(x_cp, np.flip(channel_ir))     # Flips second argument
received_fft = F@received[chanel_length-1:-chanel_length]
equalized = np.real(received_fft*channel_fr)    # In theory this should have to be the conjugate of the channel?
equalized /= np.abs(equalized)
error = np.sum(equalized != s)