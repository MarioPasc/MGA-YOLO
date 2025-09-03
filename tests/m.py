total_epochs = 500
import numpy as np
offset = total_epochs//4
print(np.arange(offset, total_epochs+offset, step=offset))