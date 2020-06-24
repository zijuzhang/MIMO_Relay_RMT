#   About:
#   A simulation showing the impact of water-filling as N -> inf
#   Takeaways:
#   The analytic results are close except for the case in which the asymptotic assumption no longer holds.


from src.stieltjes_eqn import *
import matplotlib.pyplot as plt
from src.lin_alg import *
from src.network_simulation_tools import *
import seaborn

#   Initialize System Parameters
size = 100
#   Plot Theoretical Distribution and find corresponding capacity

equal_capacity_list = []
water_capacity_list = []
size_list = [5, 10, 20, 100]


for size in size_list:

    #   Now check histogram of simulations
    irs = []
    irs_svd = []
    equal = []
    water = []
    cross_sum = 0
    main_sum = 0
    average = 10
    for i in range(average):
        G = c_rand(size, size)
        GG = G @ hermetian(G)
        water.append(capacity_water_filled(water_filling(G, 1)))
        equal.append(capacity(GG, 1/size))

#   Compare Estimated to True Capacity
    water_capacity_list.append(np.average(water))
    equal_capacity_list.append(np.average(equal))



#   Plot results
fig, ax = plt.subplots()
# plt.title("Capacity Vs. Line of Sight Rank")
plt.title("Capacity Vs. IRS Path Attenuation")
ax.plot(size_list, water_capacity_list, label='Capacity', c='g')
ax.plot(size_list, equal_capacity_list, label='Equal Power', c='r')
ax.set_ylabel('Capacity (Bits)')
ax.set_xlabel('Number of Antennas at Tx/Rx')
# ax.set_xlabel('Line of Sight Channel Rank')
plt.legend(loc="upper left")
ax.grid(True, which='both')
seaborn.despine(ax=ax, offset=0)
plt.show()