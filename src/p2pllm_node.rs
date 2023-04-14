use libp2p::{identity::Keypair, PeerId};
use std::sync::Arc;
use tch::{CModule, Device, Tensor};

pub struct P2pllmNode {
    keypair: Keypair,
    peer_id: PeerId,
    model: Arc<CModule>,
}

impl P2pllmNode {
    pub fn new(model_id: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize libp2p
        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());

        // Initialize tch-rs
        let model = tch::CModule::load(model_id)?;
        let model = Arc::new(model);

        // Create P2pllmNode
        let node = P2pllmNode {
            keypair,
            peer_id,
            model,
        };

        Ok(node)
    }

    pub fn make_prediction(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        let input_tensor = Tensor::of_slice(&input.as_bytes())
            .to(Device::Cpu)
            .to_kind(tch::Kind::Float);

        let output_tensor = self
            .model
            .forward_is(&[input_tensor])?
            .softmax(-1, tch::Kind::Float);

        // Find the index of the maximum value
        let (max_prob, max_index) = output_tensor.max(1).tuple();

        // Convert the index to a string representation
        let prediction = max_index.int64_value(&[]).to_string();

        Ok(prediction)
    }
}
