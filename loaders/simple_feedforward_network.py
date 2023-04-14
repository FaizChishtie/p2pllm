import torch
import torch.nn as nn
import sys
import struct

class SimpleFeedforwardNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleFeedforwardNetwork, self).__init__()
        self.input_layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.input_layer(x)

def load_binary_weights(file_path, model):
    with open(file_path, 'rb') as f:
        # Read input layer weights
        input_weights = struct.unpack('f' * 6, f.read(6 * 4))
        model.input_layer.weight.data = torch.tensor(input_weights).view(2, 3)

        # Read input layer biases
        input_biases = struct.unpack('f' * 2, f.read(2 * 4))
        model.input_layer.bias.data = torch.tensor(input_biases)

    return model

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input_binary_file output_torchscript_file")
        sys.exit(1)

    input_binary_file = sys.argv[1]
    output_torchscript_file = sys.argv[2]

    # Initialize the model
    model = SimpleFeedforwardNetwork(2, 2)

    # Load binary weights into the model
    model = load_binary_weights(input_binary_file, model)

    # Convert the model to TorchScript format
    torchscript_model = torch.jit.script(model)

    # Save the TorchScript model
    torch.jit.save(torchscript_model, output_torchscript_file)

if __name__ == "__main__":
    main()
