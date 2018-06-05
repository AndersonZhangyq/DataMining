import matplotlib.pyplot as plt
import math

sum = 0
data = []
for i in range(50):
    sum += math.sin(math.pi * i / 6)
    data.append(sum)

fig, ax = plt.subplots()
ax.plot(list(range(50)), data, 'o')
plt.show()
