# MouseSD-1 🎨⚡

**MouseSD-1** is a lightweight, modular local diffusion pipeline engineered for efficient image generation, fine-tuning, and inference on consumer hardware. Built with a focus on self-contained architecture and cross-platform acceleration (including Apple Silicon MPS and NVIDIA CUDA).

---

## ✨ Key Features

* **Cross-Platform Acceleration:** Native support for PyTorch backends, optimized for CUDA on Linux/Windows and Apple Silicon (`mps`) on macOS.
* **Low-Memory Footprint:** Memory-efficient attention and gradient accumulation routines designed to run smoothly on local setups.
* **Hugging Face Integration:** Streamlined serialization and automated checkpoint syncing with Hugging Face repositories.
* **Modular Pipeline:** Clean, dependency-light Python codebase separating inference, model weighting, and sampling loops.

---

## 🛠️ Project Structure

```text
MouseSD-1/
├── assets/           # Sample outputs and documentation media
├── models/           # Architecture definitions and weight configurations
├── scripts/          # Inference and training execution scripts
├── utils/            # Helper modules (logging, tensor transformations, HW acceleration)
├── requirements.txt  # Project dependencies
└── README.md
