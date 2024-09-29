# Batchify SAM processing
- Our goal is set to processing each data image in a batchfiy way.
- The prompt will be point promt, and segment everything out from a scene

# Batchify encoder
- Already doen in SC_LATENT

# Batchify prompt encoder
- Dose not need, since prompt is generated according to a 64*64 points and we make it static

# Batchfy mask decoder
- We need it