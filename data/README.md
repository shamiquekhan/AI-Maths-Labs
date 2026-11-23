# Data Directory

This directory contains datasets and corpora used by the FROM ZERO experiments.

## Structure

```
data/
├── corpora/
│   └── tiny_shakespeare.txt    # Public domain Shakespeare corpus
└── MNIST/                      # Downloaded by torchvision (gitignored)
```

## Datasets

### Tiny Shakespeare (`corpora/tiny_shakespeare.txt`)

- **Source:** Andrej Karpathy's `char-rnn` repository / `nanoGPT`
- **License:** Public domain (Shakespeare's works)
- **Size:** ~1,115,394 characters
- **Vocabulary:** 65 unique characters
- **Usage:** Character-level language modeling for MiniGPT

The file is included in the repository because it is small and public domain.

### MNIST (`MNIST/`)

- **Source:** Yann LeCun, Corinna Cortes, Christopher Burges
- **License:** Custom (attribution required for redistribution)
- **Size:** 60,000 training + 10,000 test images (28×28 grayscale)
- **Usage:** Diffusion model training (DDPM)

**Not committed to git.** The directory is created automatically when you run the diffusion training script, which downloads via `torchvision.datasets.MNIST`.

```bash
# This will download MNIST automatically
python scripts/train_diffusion.py
```

If you need to use MNIST offline, pre-download it:

```python
import torchvision.datasets as datasets

datasets.MNIST(root="data", train=True, download=True)
datasets.MNIST(root="data", train=False, download=True)
```

## Synthetic Data

The following datasets are generated programmatically (no files stored):

- **Linearly separable:** Two Gaussian clusters with configurable noise
- **Moons:** Two interleaving half-circles (non-linearly separable)
- **Synthetic shapes:** Geometric patterns for diffusion fallback

See `src/from_zero/core/datasets.py` and `src/from_zero/diffusion/data.py` for generation code.

## Adding New Data

1. **Small public domain text:** Add to `corpora/` and update this README.
2. **Large datasets:** Download in training scripts via `torchvision`/`torchaudio`/etc. Do not commit.
3. **Synthetic data:** Implement in the relevant module's `data.py`.

## License Compliance

| Dataset | License | Redistribution | In Repo? |
|---------|---------|----------------|----------|
| Tiny Shakespeare | Public domain | Yes | ✅ |
| MNIST | Custom (attribution) | Check terms | ❌ (downloaded) |
| Synthetic | MIT (code) | Yes | N/A (generated) |

**Always verify the current license terms before redistributing any dataset.**