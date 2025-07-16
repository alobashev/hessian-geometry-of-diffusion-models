import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
save_folder = 'equidistant_points_np'
save_plots = False
conv_checkpoint = "conv_model_diff_clip.pt"

# Define the neural network model (keep original)
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = torch.relu(self.norm1(self.fc1(x)))
        x = torch.relu(self.norm2(self.fc2(x)))
        x = torch.relu(self.norm3(self.fc3(x)))
        x = torch.relu(self.norm4(self.fc4(x)))
        x = self.fc5(x)
        return x

# Load model (keep original loading)
model = Net(2, 512, 1).to(device)
try:
    model.load_state_dict(torch.load(conv_checkpoint, map_location=device))
except FileNotFoundError:
    print("Error: Model file not found.")
    exit()
model.eval()

def metric_tensor(points, convex_function):
    points = points.detach().requires_grad_(True)
    outputs = -convex_function(points).sum()
    grad = torch.autograd.grad(outputs, points, create_graph=True)[0]
    
    hessian = torch.zeros(points.size(0), 2, 2, device=device, dtype=torch.float32)
    for i in range(2):
        hessian[:, i] = torch.autograd.grad(
            grad[:, i], points, torch.ones_like(grad[:, i]),
            retain_graph=True, create_graph=False
        )[0]
    return hessian.detach()

def add_equidistant_points(full_curve, num_points=10):
    """Add metric-equidistant points to the curve"""
    with torch.no_grad():
        # Create new computation graph for metric calculations
        curve = full_curve.detach().requires_grad_(True)
        
        # Compute midpoints and metric tensor with gradient tracking
        with torch.enable_grad():
            midpoints = (curve[1:] + curve[:-1])/2
            tangents = curve[1:] - curve[:-1]
            M = metric_tensor(midpoints, model)
        
        # Calculate segment lengths under the metric
        lengths = torch.sqrt(torch.einsum('ni,nij,nj->n', tangents, M.detach(), tangents.detach()))
        cum_lengths = torch.cat([torch.zeros(1, device=device), torch.cumsum(lengths, 0)])
        total_length = cum_lengths[-1]
        
        # Generate equidistant parameters
        target_lengths = torch.linspace(0, total_length, num_points, device=device)
        
        # Find corresponding points
        equidistant = []
        for tl in target_lengths:
            idx = torch.searchsorted(cum_lengths, tl) - 1
            idx = torch.clamp(idx, 0, len(curve)-2)
            t = (tl - cum_lengths[idx]) / (cum_lengths[idx+1] - cum_lengths[idx])
            equidistant.append(curve[idx] * (1-t) + curve[idx+1] * t)
            
        return torch.stack(equidistant).detach()

N_plots = 500
os.makedirs("geodesic_illustrations_with_equidistant", exist_ok=True)

for _ in range(N_plots):
    print(f"Processing {_+1}/{N_plots}")
    
    # Generate points with explicit float32 dtype
    start = torch.tensor(np.random.uniform(0.1, 0.9, 2).astype(np.float32), device=device)
    end = torch.tensor(np.random.uniform(0.1, 0.9, 2).astype(np.float32), device=device)
    
    # Curve initialization with float32
    n_points = 90
    t = torch.linspace(0, 1, n_points, device=device, dtype=torch.float32).unsqueeze(1)
    initial_curve = start * (1 - t) + end * t
    inner_points = nn.Parameter(initial_curve[1:-1].clone().detach().requires_grad_(True))
    
    optimizer = optim.Adam([inner_points], lr=0.01)
    
    # Optimization loop (keep original)
    for epoch in tqdm(range(5000), desc="Optimizing geodesic"):
        optimizer.zero_grad()
        full_curve = torch.cat([start.unsqueeze(0), inner_points, end.unsqueeze(0)])
        
        # Energy calculation
        midpoints = (full_curve[1:] + full_curve[:-1])/2
        tangents = full_curve[1:] - full_curve[:-1]
        M = metric_tensor(midpoints, model)
        energy = torch.einsum('ni,nij,nj->n', tangents, M, tangents).sum()
        
        # Continuity regularization
        tangent_diffs = tangents[1:] - tangents[:-1]
        continuity_loss = torch.norm(tangent_diffs, dim=1).square().sum()
        
        total_loss = energy + 0.1 * continuity_loss
        total_loss.backward()
        
        with torch.no_grad():
            inner_points.data = inner_points.data.clamp(1e-6, 1-1e-6)
        
        optimizer.step()
    
    # Get equidistant points
    with torch.no_grad():
        final_curve = torch.cat([start.unsqueeze(0), inner_points, end.unsqueeze(0)])
        eq_points = add_equidistant_points(final_curve, 10)
        curve_np = final_curve.cpu().numpy()
        eq_points_np = eq_points.cpu().numpy()
    
    # Plotting with equidistant points
    background = Image.open("output_grid_lower.png")
    width, height = background.size
    cropped_bg = background.crop((0, 0, width-0, height-0))

    print("Equidistant points validation:")
    print(f"Min coordinates: {eq_points_np.min(axis=0)}")
    print(f"Max coordinates: {eq_points_np.max(axis=0)}")
    print(f"NaN check: {np.isnan(eq_points_np).any()}")

    if not np.isnan(eq_points_np).any():
        if save_plots:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(cropped_bg, extent=[0,1,0,1], aspect='auto', interpolation='nearest', zorder=0)
            
            # Plot main curve and points
            ax.plot(curve_np[:, 0], curve_np[:, 1], 'white', linewidth=4.5, alpha=0.8, zorder=1, label=None)
            ax.plot(curve_np[:, 0], curve_np[:, 1], 'red', linewidth=2.0, alpha=0.8, zorder=5, label='Geodesic')
            ax.scatter(eq_points_np[:, 0], eq_points_np[:, 1], c='red', s=20, 
                       edgecolor='white', zorder=10, label='Equidistant Points')
            ax.scatter([start[0].item(), end[0].item()], [start[1].item(), end[1].item()],
                       c='red', edgecolor='white', s=80, marker='o', zorder=15, label='Endpoints')
            
            # Formatting
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.legend(loc='upper right')
            
            # Title with formatted coordinates
            start_str = np.array2string(start.cpu().numpy(), formatter={'float_kind': lambda x: "%.3f" % x})
            end_str = np.array2string(end.cpu().numpy(), formatter={'float_kind': lambda x: "%.3f" % x})
            ax.set_title(f'Geodesic from {start_str} to {end_str}\nwith Metric-Equidistant Points')
            
            # Save figure
            fname = f"geo_{start[0].item():.3f}_{start[1].item():.3f}_to_{end[0].item():.3f}_{end[1].item():.3f}.png"
            
            plt.savefig(os.path.join("geodesic_illustrations_with_equidistant", fname), bbox_inches='tight', dpi=150)
            
            plt.close()
            print("Plot is saved!")
            
        np_fname = f"geo_eq_{start[0].item():.3f}_{start[1].item():.3f}_to_{end[0].item():.3f}_{end[1].item():.3f}.npy"
        np.save(f"{save_folder}/{np_fname}", eq_points_np)

print(f"Generated {N_plots} geodesic plots with equidistant points")
