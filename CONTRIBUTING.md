# Contributing to RealRestore CLI

Thanks for your interest in contributing. This guide covers everything you need to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/199-biotechnologies/realrestore-cli
cd realrestore-cli

# Build the Rust CLI
cargo build --release

# Set up the Python inference backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt

# Run setup (downloads models)
./target/release/realrestore setup
```

## Project Structure

```
realrestore-cli/
  src/main.rs              # Rust CLI (argument parsing, output formatting)
  python/
    realrestore_cli/
      engine.py            # Core inference engine (PyTorch + diffusers)
      daemon.py            # Long-running inference daemon
      setup.py             # Model download and environment setup
      optimizations/       # MPS/MLX/quantization backends
      watermark/           # AI watermark detection and removal
      benchmarks/          # Performance measurement suite
  patches/                 # Upstream compatibility patches for MPS
  tests/                   # End-to-end and optimization tests
```

## How to Contribute

### Bug Reports

Open an issue with:
- Your hardware (chip, RAM)
- macOS version
- The full command you ran
- The error output (use `--json` for structured output)

### Pull Requests

1. Fork the repo and create a feature branch from `main`.
2. Keep changes focused. One PR per feature or fix.
3. Add tests if you touch the inference pipeline.
4. Run `cargo check` and `cargo test` before submitting.
5. Write a clear PR description explaining what changed and why.

### Areas Where Help is Welcome

- **Backend support**: CUDA, ROCm, or other GPU backends
- **New restoration tasks**: Adding support for more degradation types
- **Performance**: Faster inference on Apple Silicon (MLX optimizations, ANE support)
- **Documentation**: Usage examples, tutorials, benchmarks on different hardware
- **Testing**: More test coverage, CI improvements

## Code Style

- Rust: Follow standard `rustfmt` conventions
- Python: PEP 8, type hints where practical
- Commit messages: Short, descriptive first line. Details in the body if needed.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
