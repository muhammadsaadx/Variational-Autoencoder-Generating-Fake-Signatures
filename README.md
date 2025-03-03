# Variational Autoencoder (VAE) for Generating Fake Signatures

### Model Architecture
The code implements a Variational Autoencoder (VAE) designed to generate synthetic signatures. The VAE consists of:
- **Encoder**: The encoder network takes an input signature image and compresses it into a lower-dimensional latent representation. It outputs two vectors: the mean (μ) and the log variance (σ^2) of the latent distribution.
- **Reparameterization Trick**: To allow backpropagation through the stochastic sampling process, the reparameterization trick is used: 
  \[ z = μ + σ * ε \]
  where ε is sampled from a standard normal distribution.
- **Decoder**: The decoder takes the sampled latent vector and reconstructs a signature image from it.

### Loss Function
The training objective consists of two loss components:
- **Reconstruction Loss**: Measures the difference between the input and reconstructed images, usually using Mean Squared Error (MSE) or Binary Cross-Entropy (BCE) loss.
- **KL Divergence**: Ensures that the learned latent space follows a normal distribution, encouraging meaningful latent representations.
  \[ D_{KL}(q(z|x) || p(z)) \]
  This term helps regularize the latent space and ensures smooth generation.

### Training Process
1. Load a dataset of real signatures, normalize and preprocess the images.
2. Forward pass through the encoder to obtain the latent mean and variance.
3. Apply the reparameterization trick to sample from the latent space.
4. Decode the sampled latent vector to reconstruct the signature image.
5. Compute the loss and backpropagate to update the network weights.
6. Repeat for multiple epochs until the model generates high-quality synthetic signatures.

### Signature Generation
Once the model is trained, the decoder can be used to generate new synthetic signatures by sampling random latent vectors from a normal distribution and passing them through the decoder.

### Code Implementation
- The `train.py` script is responsible for training the VAE on a dataset of real signatures.
- The `generate.py` script loads a trained model and generates synthetic signatures by decoding random latent vectors.
- The model is implemented using PyTorch and follows a standard deep learning training pipeline.

The generated signatures are saved in the `outputs/` directory and can be evaluated visually to assess the model's performance.

