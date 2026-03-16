# MergeDNA Implementation

This is a PyTorch implementation of the [**Merge DNA**](https://arxiv.org/pdf/2511.14806) architecture. The model uses a two-stage hierarchical approach to compress DNA sequences into a latent representation, allowing it to process much longer sequences than standard transformers.

## Key Components

- **Local Encoder**: Implements window-bound token merging (Section 3.3). It uses adjacent merging within local windows to maintain the contiguous nature of DNA while achieving initial compression.
- **Latent Encoder**: Uses bipartite matching to further compress the sequence into a fixed-length latent representation for global context modeling.
- **Loss Functions (Eq. 8)**: Includes the full triple-objective training logic:
    - **MTR**: Reconstruction of the original nucleotides.
    - **Latent MTR**: Dedicated pass for training the latent representation while the local encoder remains fixed.
    - **AMTM**: Stochastic informative masking using the $1/g^2$ probability distribution (Section 3.4).
- **Positioning**: Uses Rotary Positional Embeddings (RoPE) and windowed attention for efficient long-range scaling.

## Project Structure

- `src/models/mergedna.py`: Main model assembly.
- `src/layers/token_merging.py`: Implementation of the local (adjacent) and latent (bipartite) merging operations.
- `src/training/loss.py`: Triple-loss calculation (MTR, Latent MTR, AMTM).
- `scripts/train.py`: Basic training loop with stochastic keep-ratio sampling.

## Usage

### Setup

```bash
uv sync
```

### Training

To run a test training session on synthetic DNA (20 steps):

```bash
uv run scripts/train.py --steps 20 --seq-len 256 --batch-size 4
```

**On CUDA:**
```bash
uv run scripts/train.py --steps 100 --seq-len 512 --batch-size 8 --device cuda
```
