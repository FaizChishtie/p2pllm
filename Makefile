.PHONY: build run test clean start-docker

VERSION := $(shell grep '^version\s*=\s*"' Cargo.toml | head -n 1 | cut -d'"' -f2)

# Local
build:
	cargo build --release

run: build
	./target/release/p2pllm

test:
	cargo test

clean:
	cargo clean

# Docker
build-and-start:
	docker build -t p2pllm .
	docker run -d -p 50070:50070 -p 8088:8088 -p 19888:19888 -p 9000:9000 --name p2pllm p2pllm

start:
	docker start p2pllm