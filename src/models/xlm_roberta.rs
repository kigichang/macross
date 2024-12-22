use super::activations::{HiddenAct, HiddenActLayer};
use super::auto::AutoModel;
use crate::{Dropout, Embedding, LayerNorm, Linear};
use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
    RelativeKey,
    RelativeKeyQuery,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct XLMRobertaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    #[serde(default)]
    pub position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    pub use_cache: bool,
    pub classifier_dropout: Option<f64>,
    pub model_type: Option<String>,
}

impl Default for XLMRobertaConfig {
    fn default() -> Self {
        XLMRobertaConfig {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: super::activations::HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("xlm-roberta".to_string()),
        }
    }
}

pub struct XLMRobertaEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    position_embedding_type: PositionEmbeddingType,
    padding_idx: usize,
}

impl XLMRobertaEmbeddings {
    pub fn new(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let word_embeddings = crate::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = crate::embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = crate::embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = crate::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = crate::Dropout::new(config.hidden_dropout_prob);

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
            position_embedding_type: config.position_embedding_type.clone(),
            padding_idx: config.pad_token_id,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_, seq_length) = input_ids.dims2()?;
        let position_ids = if let Some(position_ids) = position_ids {
            position_ids.clone()
        } else {
            Tensor::arange(0, seq_length as u32, input_ids.device())?
        };

        let inputs_embeds = self.word_embeddings.forward(&input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;
        let mut embeddings = (inputs_embeds + token_type_embeddings)?;

        if let PositionEmbeddingType::Absolute = self.position_embedding_type {
            let position_embeddings = self.position_embeddings.forward(&position_ids)?;
            embeddings = embeddings.broadcast_add(&position_embeddings)?;
        }

        let embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward(&embeddings)
    }
}

pub struct XLMRobertaSelfAttention {
    num_attention_heads: usize,
    attention_head_size: usize,
    // all_head_size: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    position_embedding_type: PositionEmbeddingType,
    max_position_embeddings: Option<usize>,
    distance_embedding: Option<Embedding>,
    // is_decoder: bool,
}

impl XLMRobertaSelfAttention {
    pub fn new(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        if config.hidden_size % config.num_attention_heads != 0 {
            candle_core::bail!(
                "The hidden size ({}) is not a multiple of the number of attention heads ({})",
                config.hidden_size,
                config.num_attention_heads
            );
        }

        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;

        let query = crate::linear(config.hidden_size, all_head_size, vb.pp("query"))?;
        let key = crate::linear(config.hidden_size, all_head_size, vb.pp("key"))?;
        let value = crate::linear(config.hidden_size, all_head_size, vb.pp("value"))?;
        let dropout = Dropout::new(config.attention_probs_dropout_prob);

        let position_embedding_type = config.position_embedding_type.clone();
        let (max_position_embeddings, distance_embedding) = if position_embedding_type
            == PositionEmbeddingType::RelativeKey
            || position_embedding_type == PositionEmbeddingType::RelativeKeyQuery
        {
            let max_position_embeddings = config.max_position_embeddings;
            let distance_embedding = crate::embedding(
                2 * config.max_position_embeddings - 1,
                attention_head_size,
                vb.pp("distance_embedding"),
            )?;
            (Some(max_position_embeddings), Some(distance_embedding))
        } else {
            (None, None)
        };

        Ok(Self {
            num_attention_heads,
            attention_head_size,
            // all_head_size,
            query,
            key,
            value,
            dropout,
            position_embedding_type,
            max_position_embeddings,
            distance_embedding,
        })
    }

    fn transpose_for_scores(&self, x: Tensor) -> Result<Tensor> {
        let mut new_x_shape = x.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);

        x.reshape(new_x_shape)?.transpose(1, 2)?.contiguous()
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let query_layer = self.transpose_for_scores(self.query.forward(&hidden_states)?)?;
        let key_layer = self.transpose_for_scores(self.key.forward(&hidden_states)?)?;
        let value_layer = self.transpose_for_scores(self.value.forward(&hidden_states)?)?;

        let mut attention_scores = query_layer.matmul(&key_layer.t()?.contiguous()?)?;
        attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        attention_scores = attention_scores.broadcast_add(&attention_mask)?;

        let attention_probs = candle_nn::ops::softmax(&attention_scores, D::Minus1)?;
        let attention_probs = self.dropout.forward(&attention_probs)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.permute((0, 2, 1, 3))?;
        context_layer.flatten_from(candle_core::D::Minus2)
    }
}

pub struct XLMRobertaSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl XLMRobertaSelfOutput {
    pub fn new(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let dense = crate::linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = crate::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

pub struct XLMRobertaAttention {
    self_attention: XLMRobertaSelfAttention,
    output: XLMRobertaSelfOutput,
}

impl XLMRobertaAttention {
    pub fn new(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let self_attention = XLMRobertaSelfAttention::new(vb.pp("self"), config)?;
        let output = XLMRobertaSelfOutput::new(vb.pp("output"), config)?;

        Ok(Self {
            self_attention,
            output,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let self_outputs = self.self_attention.forward(hidden_states, attention_mask)?;
        self.output.forward(&self_outputs, hidden_states)
    }
}

pub struct XLMRobertaIntermediate {
    dense: Linear,
    intermediate_act_fn: super::activations::HiddenActLayer,
}

impl XLMRobertaIntermediate {
    pub fn new(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let dense = crate::linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        let intermediate_act_fn = HiddenActLayer::new(config.hidden_act);

        Ok(Self {
            dense,
            intermediate_act_fn,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        self.intermediate_act_fn.forward(&hidden_states)
    }
}

pub struct XLMRobertaOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl XLMRobertaOutput {
    pub fn new(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let dense = crate::linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = crate::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

pub struct XLMRobertaLayer {
    // chunk_size_feed_forward: usize,
    // seq_len_dim: usize,
    attention: XLMRobertaAttention,
    // is_decoder: bool,
    // add_cross_attention: bool,
    // crossattention: Option<XLMRobertaAttention>,
    intermediate: XLMRobertaIntermediate,
    output: XLMRobertaOutput,
}

impl XLMRobertaLayer {
    pub fn new(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let attention = XLMRobertaAttention::new(vb.pp("attention"), config)?;
        let intermediate = XLMRobertaIntermediate::new(vb.pp("intermediate"), config)?;
        let output = XLMRobertaOutput::new(vb.pp("output"), config)?;

        Ok(Self {
            // seq_len_dim: 1,
            attention,
            intermediate,
            output,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        self.feed_forward_chunk(&attention_output)
    }

    fn feed_forward_chunk(&self, attention_output: &Tensor) -> Result<Tensor> {
        let intermediate_output = self.intermediate.forward(attention_output)?;
        self.output.forward(&intermediate_output, attention_output)
    }
}

pub struct XLMRobertaEncoder {
    // config: XLMRobertaConfig,
    layer: Vec<XLMRobertaLayer>,
    // gradient_checkpointing: bool,
}

impl XLMRobertaEncoder {
    pub fn new(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let layer: Vec<XLMRobertaLayer> = (0..config.num_hidden_layers)
            .map(|idx| XLMRobertaLayer::new(vb.pp(format!("layer.{idx}")), config))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            // config: config.clone(),
            layer,
            // gradient_checkpointing: false,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in &self.layer {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }
}

pub struct XLMRobertaPooler {
    dense: Linear,
    // activation: fn(&Tensor) -> Result<Tensor>,
}

impl XLMRobertaPooler {
    pub fn new(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let dense = crate::linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        // let activation = tanh;

        Ok(Self { dense })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // We "pool" the model by simply taking the hidden state corresponding
        // to the first token.
        let first_token_tensor = hidden_states.i((.., 0))?.contiguous()?;
        let pooled_output = self.dense.forward(&first_token_tensor)?;
        pooled_output.tanh()
    }
}

pub struct XLMRobertaModel {
    embeddings: XLMRobertaEmbeddings,
    encoder: XLMRobertaEncoder,
}

impl AutoModel<XLMRobertaConfig> for XLMRobertaModel {
    type Model = Self;
    fn new(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let embeddings = XLMRobertaEmbeddings::new(vb.pp("embeddings"), config)?;
        let encoder = XLMRobertaEncoder::new(vb.pp("encoder"), config)?;

        Ok(Self {
            embeddings,
            encoder,
        })
    }
}

impl XLMRobertaModel {
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids, None)?;

        let extended_attention_mask =
            self.get_extended_attention_mask(&attention_mask, DType::F32)?;

        self.encoder
            .forward(&embedding_output, &extended_attention_mask)
    }

    //
    fn get_extended_attention_mask(&self, attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
        let extended_attention_mask = match attention_mask.rank() {
            3 => attention_mask.unsqueeze(1)?,
            2 => attention_mask.unsqueeze(1)?.unsqueeze(2)?,
            _ => candle_core::bail!("wrong attention_mask (shape {:?})", attention_mask.dims()),
        };

        // Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        // masked positions, this operation will create a tensor which is 0.0 for
        // positions we want to attend and the dtype's smallest value for masked positions.
        // Since we are adding it to the raw scores before the softmax, this is
        // effectively the same as removing these entirely.
        let extended_attention_mask = extended_attention_mask.to_dtype(dtype)?; // fp16 compatibility
        (extended_attention_mask.ones_like()? - &extended_attention_mask)?
            .broadcast_mul(&Tensor::try_from(f32::MIN)?.to_device(attention_mask.device())?)
    }
}

pub struct XLMRobertaModelWithPooler {
    bert: XLMRobertaModel,
    pooler: XLMRobertaPooler,
}

impl AutoModel<XLMRobertaConfig> for XLMRobertaModelWithPooler {
    type Model = Self;
    fn new(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let bert = XLMRobertaModel::new(vb.clone(), config)?;
        let pooler = XLMRobertaPooler::new(vb.pp("pooler"), config)?;

        Ok(Self { bert, pooler })
    }
}

impl XLMRobertaModelWithPooler {
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let hidden_states = self
            .bert
            .forward(input_ids, token_type_ids, attention_mask)?;
        self.pooler.forward(&hidden_states)
    }
}

pub struct XLMRobertaLMHead {
    dense: Linear,
    layer_norm: LayerNorm,
    decoder: Linear,
    // bias: Tensor,
}

impl XLMRobertaLMHead {
    pub fn new(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let dense = crate::linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = crate::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("layer_norm"),
        )?;
        let decoder =
            crate::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("decoder"))?;
        // let bias = Tensor::zeros(&[config.vocab_size], (DType::Float, vb.device()))?;
        // decoder.set_bias(Some(bias.clone()))?;

        Ok(Self {
            dense,
            layer_norm,
            decoder,
            // bias,
        })
    }

    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let x = self.dense.forward(features)?;
        let x = x.gelu()?;
        let x = self.layer_norm.forward(&x)?;
        let x = self.decoder.forward(&x)?;
        Ok(x)
    }

    // pub fn tie_weights(&mut self) -> Result<()> {
    //     if self.decoder.bias().device().unwrap().is_meta() {
    //         self.decoder.set_bias(Some(self.bias.clone()))?;
    //     } else {
    //         self.bias = self.decoder.bias().unwrap().clone();
    //     }
    //     Ok(())
    // }
}

pub struct XLMRobertaForMaskedLM {
    roberta: XLMRobertaModel,
    lm_head: XLMRobertaLMHead,
}

impl AutoModel<XLMRobertaConfig> for XLMRobertaForMaskedLM {
    type Model = Self;
    fn new(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let roberta = XLMRobertaModel::new(vb.pp("roberta"), config)?;
        let lm_head = XLMRobertaLMHead::new(vb.pp("lm_head"), config)?;

        Ok(Self { roberta, lm_head })
    }
}

impl XLMRobertaForMaskedLM {
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let sequence_output = self
            .roberta
            .forward(input_ids, token_type_ids, attention_mask)?;

        self.lm_head.forward(&sequence_output)
    }

    pub fn get_output_embeddings(&self) -> &Linear {
        &self.lm_head.decoder
    }

    pub fn set_output_embeddings(&mut self, new_embeddings: Linear) {
        self.lm_head.decoder = new_embeddings;
    }
}
