import torch
import torch.nn.functional as F

from tqdm import tqdm

def extract(a, t, x_shape):
    # Takes a data tensor a and an index tensor t, and returns a new tensor
    # whose i^th element is just a[t[i]]. Note that this will be useful when
    # we would want to choose the alphas or betas corresponding to different
    # indices t's in a batched manner without for loops.
    # Inputs:
    #   a: Tensor, generally of shape (batch_size,)
    #   t: Tensor, generally of shape (batch_size,)
    #   x_shape: Shape of the data, generally (batch_size, 3, 32, 32)
    # Returns:
    #   out: Tensor of shape (batch_size, 1, 1, 1) generally, the number of 1s are
    #         determined by the number of dimensions in x_shape.
    #         out[i] contains a[t[i]]
    
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def alphas_betas_sequences_helper(beta_start, beta_end, T):
    def linear_beta_schedule(beta_start, beta_end, timesteps):
        return torch.linspace(beta_start, beta_end, timesteps)

    betas = linear_beta_schedule(beta_start, beta_end, T)                        # WRITE CODE HERE: Define the linear beta schedule
    alphas = 1. - betas                             # WRITE CODE HERE: Compute the alphas as 1 - betas
    sqrt_recip_alphas = torch.sqrt(1. / alphas)                # WRITE CODE HERE: Returns 1/square_root(\alpha_t)
    alphas_cumprod = torch.cumprod(alphas, dim=0)                # WRITE CODE HERE: Compute product of alphas up to index t, \bar{\alpha}
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)              # WRITE CODE HERE: Returns sqaure_root(\bar{\alpha}_t)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)    # WRITE CODE HERE: Returns square_root(1 - \bar{\alpha}_t)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1)             # WRITE CODE HERE: Right shifts \bar{\alpha}_t; with first element as 1.
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)               # WRITE CODE HERE: Contains the posterior variances $\tilde{\beta}_t$

    return betas, alphas, sqrt_recip_alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, alphas_cumprod_prev, posterior_variance

# betas, alpha, sqrt_recip_alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, alphas_cumprod_prev, posterior_variance = alphas_betas_sequences_helper()

def q_sample(x_start, t, coefficients, noise=None):
    # Forward Diffusion Sampling Process
    # Inputs:
    #   x_start: Tensor of original images of size (batch_size, 3, 32, 32)
    #   t: Tensor of timesteps, of shape (batch_size,)
    #   noise: Optional tensor of same shape as x_start, signifying that the noise to add is already provided.
    #   coefficients: 2-tuple
    # Returns:
    #   x_noisy: Tensor of noisy images of size (batch_size, 3, 32, 32)
    #             x_noisy[i] is sampled from q(x_{t[i]} | x_start[i])
    
    if noise is None:
      noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(coefficients[0], t, x_start.shape)          # WRITE CODE HERE: Obtain the cumulative product sqrt_alphas_cumprod up to a given point t in a batched manner for different t's
    sqrt_one_minus_alphas_cumprod_t = extract(coefficients[1], t, x_start.shape) # WRITE CODE HERE: Same as above, but for sqrt_one_minus_alphas_cumprod
                          # WRITE CODE HERE: Given the above co-efficients and the noise, generate a noisy sample based on q(x_t | x_0)
    
    x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    return x_noisy

def p_sample(model, x, t, t_index, coefficients,  noise=None):
    # Given the denoising model, batched input x, and time-step t, returns a slightly denoised sample at time-step t-1
    # Inputs:
    #   model: The denoising (parameterized noise) model
    #   x: Batched noisy input at time t; size (batch_size, 3, 32, 32)
    #   t: Batched time steps; size (batch_size,)
    #   t_index: Single time-step, whose batched version is present in t
    #   coefficients: 4-tuple
    # Returns:
    #   sample: A sample from the distribution p_\theta(x_{t-1} | x_t); mode if t=0
    with torch.no_grad():
        betas_t = extract(coefficients[0], t, x.shape) # WRITE CODE HERE: Similar to q_sample, extract betas for specific t's
        sqrt_one_minus_alphas_cumprod_t = extract(coefficients[1], t, x.shape) # WRITE CODE HERE: Same as above, but for sqrt_one_minus_alphas_cumprod
        sqrt_recip_alphas_t = extract(coefficients[2], t, x.shape)           # WRITE CODE HERE: Same as above, but for sqrt_recip_alphas

        p_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)  # WRITE CODE HERE: Obtain the mean of the distribution p_\theta(x_{t-1} | x_t)

        if t_index == 0:
            sample = p_mean                     # WRITE CODE HERE: Set the sample as the mode
        else:
            posterior_variance_t =  extract(coefficients[3], t, x.shape)         # WRITE CODE HERE: Same as betas_t, but for posterior_variance
            # WRITE CODE HERE
            # Generate a sample from p_\theta(x_{t-1} | x_t) by generating some noise or if available taking in the noise given
            # Followed by reparameterization to obtain distribution from the mean and variance computed above.
            if noise is None:
                noise = torch.randn_like(x)
            sample = p_mean + torch.sqrt(posterior_variance_t) * noise

        return sample

def p_sample_loop(model, shape, timesteps, T, coefficients, noise=None):
    # Given the model, and the shape of the image, returns a sample from the data distribution by running through the backward diffusion process.
    # Inputs:
    #   model: The denoising model
    #   shape: Shape of the samples; set as (batch_size, 3, 32, 32)
    #   noise: (timesteps+1, batch_size, 3, 32, 32)
    # Returns:
    #   imgs: Samples obtained, as well as intermediate denoising steps, of shape (T, batch_size, 3, 32, 32)
    with torch.no_grad():
        b = shape[0]
        # Start from pure noise (x_T)
        img = torch.randn(shape, device=model.device) if noise is None else noise[0]
        imgs = []
        
        for i in tqdm(reversed(range(0, timesteps)), desc='Sampling', total=T, leave=False):
            img = p_sample(model=model, x=img, t=torch.full((b,), i, device=model.device, dtype=torch.long), t_index=i, coefficients=coefficients, noise=noise[i+1])
            imgs.append(img.cpu())
        
        return torch.stack(imgs)

def p_losses(denoise_model, x_start, t, coefficients, noise=None):
    # Returns the loss for training of the denoise model
    # Inputs:
    #   denoise_model: The parameterized model
    #   x_start: The original images; size (batch_size, 3, 32, 32)
    #   t: Timesteps (can be different at different indices); size (batch_size,)
    # Returns:
    #   loss: Loss for training the model
    noise = torch.randn_like(x_start) if noise is None else noise
    
    x_noisy = q_sample(x_start, t, coefficients, noise)        # WRITE CODE HERE: Obtain the noisy image from the original images x_start, at times t, using the noise noise.
    predicted_noise = denoise_model(x_noisy, t) # WRITE CODE HERE: Obtain the prediction of the noise using the model.
    
    loss = torch.nn.functional.huber_loss(predicted_noise, noise, delta=1)           # WRITE CODE HERE: Compute the huber loss between true noise generated above, and the noise estimate obtained through the model.
    
    return loss


def t_sample(timesteps, batch_size, device):
    # Returns randomly sampled timesteps
    # Inputs:
    #   timesteps: The max number of timesteps; T
    #   batch_size: batch_size used in training
    # Returns:
    #   ts: Tensor of size (batch_size,) containing timesteps randomly sampled from 0 to timesteps-1
    
    ts = torch.randint(0, timesteps, (batch_size,), device=device)   # WRITE CODE HERE: Randommly sample a tensor of size (batch_size,) where entries are independently sampled from [0, ..., timesteps-1]()
    return ts