import numpy as np
from tqdm import tqdm


DATASETS_PATH = './data/' #CHANGE HERE

states_clip_embeddings = np.load(DATASETS_PATH+"states_clip_embeddings.npy",)
labels = np.load(DATASETS_PATH+"labels.npy",)

prob_grid_list = []

for idx in tqdm(range(len(states_clip_embeddings)), position=0, leave=True):
    dists = np.sqrt(2*(1.00001 - states_clip_embeddings @ states_clip_embeddings[idx]))
    dists = np.exp(-10*dists)
    idxes = np.argsort(dists)[::1]
    
    # Create probability density grid 
    def create_probability_grid(labels, dists, idxes, grid_size=128):
        # Extract coordinates and values
        x_coords = labels[idxes, 0]
        y_coords = 1 - labels[idxes, 1]  # Flip y-axis
        values = dists[idxes]
    
        # Scale coordinates to grid indices
        x_scaled = x_coords * (grid_size - 1)
        y_scaled = y_coords * (grid_size - 1)
    
        # Get integer parts and fractions
        i = np.floor(x_scaled).astype(int)
        j = np.floor(y_scaled).astype(int)
        dx = x_scaled - i
        dy = y_scaled - j
    
        # Clip indices to stay within grid bounds
        i = np.clip(i, 0, grid_size-2)
        j = np.clip(j, 0, grid_size-2)
    
        # Calculate weights for bilinear interpolation
        w11 = (1 - dx) * (1 - dy)
        w21 = dx * (1 - dy)
        w12 = (1 - dx) * dy
        w22 = dx * dy
    
        # Create empty grid
        grid = np.zeros((grid_size, grid_size))
    
        # Accumulate values using vectorized operations
        np.add.at(grid, (j, i), w11 * values)
        np.add.at(grid, (j, i+1), w21 * values)
        np.add.at(grid, (j+1, i), w12 * values)
        np.add.at(grid, (j+1, i+1), w22 * values)
    
        grid = np.clip(grid,0,1)
    
        # Normalize to create probability density
        grid /= grid.sum()
        return grid[::-1,:]
    
    # Create the probability grid
    prob_grid = create_probability_grid(labels, dists, idxes)
    prob_grid_list.append(prob_grid)

posterior_distributions = np.stack(prob_grid_list)[:,None,:,:]

np.save(DATASETS_PATH+"posterior_distributions.npy", posterior_distributions)