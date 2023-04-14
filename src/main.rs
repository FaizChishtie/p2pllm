use std::process::Command;
use structopt::StructOpt;

mod hdfs_utils;
mod p2pllm_node;

use hdfs_utils::HdfsUtils;
use p2pllm_node::P2pllmNode;

#[derive(StructOpt, Debug)]
#[structopt(name = "p2pllm")]
enum P2pllm {
    Upload {
        #[structopt(short, long)]
        file: String,
        #[structopt(short, long)]
        dest: String,
        #[structopt(short, long)]
        loader: String,
    },
    Connect {
        #[structopt(short, long)]
        model_id: String,
    },
    Predict {
        #[structopt(short, long)]
        model_id: String,
        #[structopt(short, long)]
        input: String,
    },
}

fn main() {
    let opt = P2pllm::from_args();

    // Initialize HdfsUtils with the HDFS URI.
    let hdfs_utils = HdfsUtils::new("hdfs://localhost:9000").expect("Failed to connect to HDFS");

    match opt {
        P2pllm::Upload { file, dest, loader } => {
            // Create an HdfsUtils instance
            let hdfs_utils = HdfsUtils::new("hdfs://localhost:9000")?;

            // Upload the file to HDFS
            hdfs_utils.upload_file(&file, &dest)?;

            let model_id = "simple_feedforward.bin";

            // Convert the uploaded binary file to a TorchScript model using the specified loader script
            let loader_script_path = format!("./loaders/{}", loader);
            let torchscript_file_path = format!("{}/{}.pt", dest, model_id);
            let output = Command::new("python")
                .arg(loader_script_path)
                .arg(&file)
                .arg(&torchscript_file_path)
                .output()?;
            println!("Conversion output: {:?}", output);

            println!("File uploaded and converted to TorchScript model.");
        }
        P2pllm::Connect { model_id } => {
            // Create a new p2pllm node and connect to the distributed LLM.
            let node = P2pllmNode::new(&model_id).expect("Failed to connect to distributed LLM");

            // Interact with the LLM here.
        }
        P2pllm::Predict { model_id, input } => {
            let node = P2pllmNode::new(&model_id).expect("Failed to connect to distributed LLM");
            let prediction = node
                .make_prediction(&input)
                .expect("Failed to make prediction");
            println!("Prediction: {}", prediction);
        }
    }
}
