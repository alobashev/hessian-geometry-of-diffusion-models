import os
import torch
import clip
import numpy as np
from tqdm import tqdm

# Setup paths
image_dir = 'grid_100_100/images/' 
embedding_dir = 'grid_100_100/embeddings/' #where to store embeddings
os.makedirs(embedding_dir, exist_ok=True)

# Initialize CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

model.eval()  
def process_images():
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.npy'))]
    
    for filename in tqdm(image_files, desc="Processing images"):
        try:
            # Load and preprocess image
            image = np.load(os.path.join(image_dir, filename), allow_pickle=True).item()['image'][0]
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            # Get CLIP embedding
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                embedding = image_features.cpu().numpy()
            
            # Save embedding
            base_name = os.path.splitext(filename)[0]
            np.save(os.path.join(embedding_dir, f"{base_name}.npy"), embedding)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

if __name__ == "__main__":
    process_images()
    print("Embedding generation complete!")