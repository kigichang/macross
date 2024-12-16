use super::activations::HiddenAct;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    #[default]
    Rope,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct NTKScaling {
    pub factor: f64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum RopeScaling {
    Ntk(NTKScaling),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    pub hidden_act: HiddenAct,
    intermediate_size: usize,
    hidden_dropout_prob: f64,
    attention_probs_dropout_prob: f64,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    initializer_range: f64,
    layer_norm_type: String,
    layer_norm_eps: f64,
    #[serde(default)]
    position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    use_cache: bool,
    rope_theta: f64,
    rope_scaling: Option<RopeScaling>,
    classifier_dropout: Option<f64>,

    // #[serde(default)]
    // pack_qkv: bool,
    // #[serde(default)]
    // unpad_inputs: bool,
    // use_memory_efficient_attention: bool,
    logn_attention_scale: bool,
    logn_attention_clip1: bool,

    id2label: Option<HashMap<String, String>>,
    model_type: Option<String>,
}
