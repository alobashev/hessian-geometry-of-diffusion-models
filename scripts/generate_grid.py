import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from tqdm import tqdm 
import argparse
import os
torch.manual_seed(3298878210511803462)

class DDIM(StableDiffusionPipeline):

    # Sample function (regular DDIM)
    @torch.no_grad()
    def sample(self, prompt, start_step=0, start_latents=None,
               guidance_scale=5, num_inference_steps=50,
               num_images_per_prompt=1, do_classifier_free_guidance=True,
               negative_prompt='', device='cuda', eta=0):
      
        # Encode prompt
        text_embeddings = self._encode_prompt(
                prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        
        # Set num inference steps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
    
        # Create a random starting point if not provided
        if start_latents is None:
            start_latents = torch.randn(1, 4, 64, 64, device=device)
            start_latents *= self.scheduler.init_noise_sigma
    
        latents = start_latents.clone()
    
        for i in range(start_step, num_inference_steps):
        
            t = self.scheduler.timesteps[i]
    
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    
            # Predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    
            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
            # Normally we'd rely on the scheduler to handle the update step:
            # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    
            # Instead, let's do it ourselves:
            prev_t = max(1, t.item() - (1000//num_inference_steps)) # t-1
            alpha_t = self.scheduler.alphas_cumprod[t.item()]
            alpha_t_prev = self.scheduler.alphas_cumprod[prev_t]
            predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
            direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred

            # Compute the variance for stochastic sampling
            c1 = (1 - alpha_t_prev).sqrt() * eta
            c2 = (1 - alpha_t_prev).sqrt() * ((1 - eta ** 2) ** 0.5) 

            noise = torch.randn_like(latents) if ( eta > 0 and i < num_inference_steps-1 ) else torch.zeros_like(latents)
            
            latents = alpha_t_prev.sqrt()*predicted_x0  + c1 * noise + c2 * noise_pred
    

        # Post-processing
        images = self.decode_latents(latents)
        images = self.numpy_to_pil(images)
        
        # Return the final images along with the history of latents and predicted x0
        return images


def interpolated_latent(latent_0, latent_1, latent_2, alpha, beta, noise=1e-6):
    """Interpolates between three latent vectors with Gaussian noise."""
    interpolated_latent = latent_0 + alpha * (latent_1 - latent_0) + beta * (latent_2 - latent_0) + torch.randn_like(latent_0) * noise
    return interpolated_latent / interpolated_latent.std()
    



def save_interpolations(latent_0, latent_1, latent_2, num_samples, eta, noise_level, folder_name, prompt, negative_prompt, pipe, device):
    """
    Generates and saves interpolated images, latents, and predicted x0 latents as .npy files.

    Args:
        latent_0: First latent vector.
        latent_1: Second latent vector.
        latent_2: Third latent vector.
        num_samples: Number of interpolation grid points.
        eta: Eta parameter for DDIM sampling.
        noise_level: Standard deviation of the Gaussian noise.
        folder_name: Base folder for saving the results.
        prompt: Text prompt for generation.
        negative_prompt: Negative text prompt for guidance.
        pipe: DDIM pipeline object.
        device: Device to run the model on.
    """
    # Create subfolders
    os.makedirs(os.path.join(folder_name, "images"), exist_ok=True)

    alpha_values = np.linspace(0, 1, num_samples)
    beta_values = np.linspace(0, 1, num_samples)
    total_iterations = len(alpha_values) * len(beta_values)
    
    # Create single progress bar
    with tqdm(total=total_iterations, desc="Generating interpolations") as pbar:
        for i in range(len(alpha_values)):
            for j in range(len(beta_values)):
                # Generate interpolated latent
                latent = interpolated_latent(latent_0, latent_1, latent_2, 
                                           alpha_values[i], beta_values[j], 
                                           noise=noise_level)
                
                # Generate images and latents
                imgs = pipe.sample(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    start_latents=latent,
                    device=device, 
                    eta=eta,
                )
                
                hash_id = f"a{alpha_values[i]:.4f}_b{beta_values[j]:.4f}_i{i:03d}_j{j:03d}"
                # Save results
                save_path = os.path.join(folder_name, f"images/sample_{hash_id}.npy")
                np.save(save_path, {"image": imgs, 
                                  "alpha": alpha_values[i], 
                                  "beta": beta_values[j]})
                
                # Update progress bar
                pbar.update(1)
            

    print(f"Interpolations saved to {folder_name}")

if __name__ == "__main__":
    # Argument parsing setup
    parser = argparse.ArgumentParser(description="Generate interpolations between three latents with noise.")
    parser.add_argument("--num_samples", type=int, default=100,
                    help="Number of interpolation grid points (both alpha and beta will be linspace of this length).")
    parser.add_argument("--noise_level", type=float, default=0.0, help="Noise level to apply during interpolation")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta (stochasticity control)")
    parser.add_argument("--latents_file", type=str, default='stable_version_latents.pt', help="Path to torch anchors latent file") #CHANGE HERE
    parser.add_argument("--output_folder", type=str, default='./', help="Where to save generated images")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device to use")
    parser.add_argument("--prompt", type=str, default='High quality picture, 4k, detailed', help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default='blurry, ugly, stock photo', help="Negative text prompt for generation")
    args = parser.parse_args()

    # Load the DDIM pipeline
    path = "realisticVisionV60B1_v51VAE.safetensors" #change here to your model path
    pipe = DDIM.from_single_file(path)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(args.device)
    # pipe.load_lora_weights("smooth_lora.safetensors") #if you want to use a lora, uncomment this line and change the path

    # Set random seed for reproducibility
    torch.manual_seed(857862754)

    # Create output folder
    subfolder = os.path.join(
        args.output_folder,
        f"grid_100_100" # You can change this to any name you want
    )
    os.makedirs(subfolder, exist_ok=True)

    # Load latents
    start_latents = torch.load(args.latents_file).to(args.device)
    latent_0 = start_latents[0]
    latent_1 = start_latents[1]
    latent_2 = start_latents[2]

    print(f"Generating interpolations ...")
    save_interpolations( #latent_0, latent_1, latent_2, num_samples, eta, noise_level, folder_name, prompt, negative_prompt, pipe, device
        latent_0=latent_0, latent_1=latent_1, latent_2=latent_2,
        num_samples=args.num_samples, noise_level=args.noise_level, eta=args.eta, folder_name=subfolder,
        prompt=args.prompt, negative_prompt=args.negative_prompt, pipe=pipe, device=args.device
    )