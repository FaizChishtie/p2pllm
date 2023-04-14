# Stage 1: Build p2pllm in a Rust container with Hadoop and LibTorch libraries
FROM rust:latest as builder

# Install dependencies
RUN apt-get update && \
  apt-get install -y wget

# Download and extract OpenJDK 11
RUN wget https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz && \
  tar -xzf openjdk-11.0.2_linux-x64_bin.tar.gz && \
  rm openjdk-11.0.2_linux-x64_bin.tar.gz

# Set JAVA_HOME environment variable
ENV JAVA_HOME /jdk-11.0.2
ENV PATH $JAVA_HOME/bin:$PATH

# Set HADOOP_HOME environment variable
ENV HADOOP_HOME /opt/hadoop

# Install Hadoop
ARG HADOOP_VERSION=3.3.1
RUN curl -O https://downloads.apache.org/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz && \
  tar -xzf hadoop-${HADOOP_VERSION}.tar.gz -C /opt && \
  rm hadoop-${HADOOP_VERSION}.tar.gz && \
  ln -s /opt/hadoop-${HADOOP_VERSION} ${HADOOP_HOME}

# Add Hadoop libraries to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH ${HADOOP_HOME}/lib/native:${LD_LIBRARY_PATH}

# Download and install LibTorch
ARG LIBTORCH_VERSION=1.10.0
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip && \
  unzip libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip -d /opt && \
  rm libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip

# Set LIBTORCH_HOME environment variable
ENV LIBTORCH_HOME /opt/libtorch
ENV LD_LIBRARY_PATH ${LIBTORCH_HOME}/lib:${LD_LIBRARY_PATH}

# Copy the p2pllm source code
COPY . /p2pllm
WORKDIR /p2pllm

# Build the p2pllm binary
RUN C_INCLUDE_PATH=$JAVA_HOME/include:$JAVA_HOME/include/linux:$LIBTORCH_HOME/include:$LIBTORCH_HOME/include/torch/csrc/api/include \
  LIBRARY_PATH=$LIBTORCH_HOME/lib \
  cargo build --release

# Stage 2: Copy the compiled binary to a Hadoop container
FROM apache/hadoop:3

# Copy the p2pllm binary from the builder stage
COPY --from=builder /p2pllm/target/release/p2pllm /p2pllm

# Set the PATH to include the p2pllm binary
ENV PATH="/p2pllm:${PATH}"

# Run the Hadoop services
CMD ["/bin/bash", "-c", "service ssh start && /etc/bootstrap.sh"]