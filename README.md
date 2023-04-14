# p2pllm

p2pllm is a distributed Large Language Model (LLM) system that leverages HDFS, libp2p, and other Rust libraries to enable parallel LLM processing and P2P file sharing across multiple computers.

## Features

- Distributed file system using HDFS
- P2P file sharing using libp2p
- LLM parallelization using tch-rs (PyTorch C++ API)
- Task scheduling and load balancing with Tokio
- Fault tolerance and data redundancy using Sled
- Secure communication with rustls and rust-crypto

## Prerequisites

- A properly configured Hadoop cluster with HDFS
- Rust toolchain (stable)
- HDFS client installed on participating computers

## Installation

### Binary

1. Clone the p2pllm repository:

```bash
git clone https://github.com/faizchishtie/p2pllm.git
cd p2pllm
```

2. Build the p2pllm project:
```bash
cargo build --release
```

The p2pllm binary will be available in the target/release directory.

### Docker

1. Clone the p2pllm repository:

```bash
git clone https://github.com/faizchishtie/p2pllm.git
cd p2pllm
```

> Note: you can run `make build-and-start` to do the following.

2. Build the Docker image:

```bash
docker build -t p2pllm .
```

3. Run the Docker container:

```bash
docker run -d -p 50070:50070 -p 8088:8088 -p 19888:19888 -p 9000:9000 --name p2pllm p2pllm
```

## Usage

To upload a binary file containing model weights, you'll need to provide a loader script that converts the binary file to a TorchScript format. You can find instructions on creating and submitting loader parsers in the [Loaders](LOADERS.md) file.

### Uploading LLM binary file

```bash
./p2pllm upload -f /path/to/llm_binary -d /destination/on/hdfs
```

### Connecting to a distributed LLM

```bash
./p2pllm connect -m model_id
```

### Interacting with the LLM

Making predictions

```bash
./p2pllm predict -m model_id -i "input text"
```
