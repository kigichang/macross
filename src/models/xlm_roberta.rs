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
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
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
            create_position_ids_from_input_ids(input_ids, self.padding_idx as u32, 1)?
            //Tensor::arange(0, seq_length as u32, input_ids.device())?
        };

        let inputs_embeds = self.word_embeddings.forward(&input_ids)?;
        // println!("inputs_embeds: {:?}", inputs_embeds.shape());
        // crate::print_tensor::<f32>(&inputs_embeds)?;

        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;
        // println!("token_type_embeddings: {:?}", token_type_embeddings.shape());
        // crate::print_tensor::<f32>(&token_type_embeddings)?;

        let mut embeddings = (inputs_embeds + token_type_embeddings)?;
        // println!("embeddings: {:?}", embeddings.shape());
        // crate::print_tensor::<f32>(&embeddings)?;

        if let PositionEmbeddingType::Absolute = self.position_embedding_type {
            // println!("position_ids: {:?}", position_ids.shape());
            // crate::print_tensor::<u8>(&position_ids)?;
            let position_embeddings = self.position_embeddings.forward(&position_ids)?;
            // println!("position_embeddings: {:?}", position_embeddings.shape());
            // crate::print_tensor::<f32>(&position_embeddings)?;

            embeddings = embeddings.broadcast_add(&position_embeddings)?;
            // println!("embeddings: {:?}", embeddings.shape());
            // crate::print_tensor::<f32>(&embeddings)?;
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
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
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
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
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
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let self_attention = XLMRobertaSelfAttention::load(vb.pp("self"), config)?;
        let output = XLMRobertaSelfOutput::load(vb.pp("output"), config)?;

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
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
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
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
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
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let attention = XLMRobertaAttention::load(vb.pp("attention"), config)?;
        let intermediate = XLMRobertaIntermediate::load(vb.pp("intermediate"), config)?;
        let output = XLMRobertaOutput::load(vb.pp("output"), config)?;

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
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let layer: Vec<XLMRobertaLayer> = (0..config.num_hidden_layers)
            .map(|idx| XLMRobertaLayer::load(vb.pp(format!("layer.{idx}")), config))
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
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
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

impl AutoModel for XLMRobertaModel {
    type Config = XLMRobertaConfig;
    type Model = Self;
    fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let embeddings = XLMRobertaEmbeddings::load(vb.pp("embeddings"), config)?;
        let encoder = XLMRobertaEncoder::load(vb.pp("encoder"), config)?;

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
        //println!("embedding_output: {:?}", embedding_output.to_vec3::<f32>()?);
        // println!("embedding_output");
        // crate::print_tensor::<f32>(&embedding_output)?;

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

impl AutoModel for XLMRobertaModelWithPooler {
    type Config = XLMRobertaConfig;
    type Model = Self;
    fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let bert = XLMRobertaModel::load(vb.clone(), config)?;
        let pooler = XLMRobertaPooler::load(vb.pp("pooler"), config)?;

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
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let dense = crate::linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = crate::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("layer_norm"),
        )?;
        let mut decoder = crate::linear(config.hidden_size, config.vocab_size, vb.pp("decoder"))?;
        let bias = Tensor::zeros(config.vocab_size, DType::F32, vb.device())?;
        decoder.set_bias(Some(bias));

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

impl AutoModel for XLMRobertaForMaskedLM {
    type Config = XLMRobertaConfig;
    type Model = Self;
    fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let roberta = XLMRobertaModel::load(vb.pp("roberta"), config)?;
        let lm_head = XLMRobertaLMHead::load(vb.pp("lm_head"), config)?;

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

pub fn create_position_ids_from_input_ids(
    input_ids: &Tensor,
    padding_idx: u32,
    past_key_values_length: u8,
) -> Result<Tensor> {
    let mask = input_ids.ne(padding_idx)?;
    let incremental_indices = cumsum_2d(&mask, 0, input_ids.device())?;

    let incremental_indices = incremental_indices
        .broadcast_add(&Tensor::new(&[past_key_values_length], input_ids.device())?)?;

    Ok(incremental_indices)
}

fn cumsum_2d(mask: &Tensor, dim: u8, device: &candle_core::Device) -> Result<Tensor> {
    let mask = mask.to_vec2::<u8>()?;

    let rows = mask.len();
    let cols = mask[0].len();

    let mut result = mask.clone();

    match dim {
        0 => {
            // Cumulative sum along rows
            for i in 0..rows {
                for j in 1..cols {
                    result[i][j] += result[i][j - 1];
                }
            }
        }
        1 => {
            // Cumulative sum along columns
            for j in 0..cols {
                for i in 1..rows {
                    result[i][j] += result[i - 1][j];
                }
            }
        }
        _ => panic!("Dimension not supported"),
    }

    let result = Tensor::new(result, &device)?;

    Ok(result)
}

// -----------------------------------------------------------------------------

// use crate::{linear, AutoModel, Linear};
// use candle_core::{DType, Module, Result, Tensor};
// use candle_nn::{
//     embedding, layer_norm, ops::softmax_last_dim, Activation, Embedding, LayerNorm, VarBuilder,
// };

// #[derive(Debug, Clone, serde::Deserialize)]
// pub struct Config {
//     pub hidden_size: usize,
//     pub layer_norm_eps: f64,
//     pub attention_probs_dropout_prob: f32,
//     pub hidden_dropout_prob: f32,
//     pub num_attention_heads: usize,
//     pub position_embedding_type: String,
//     pub intermediate_size: usize,
//     pub hidden_act: Activation,
//     pub num_hidden_layers: usize,
//     pub vocab_size: usize,
//     pub max_position_embeddings: usize,
//     pub type_vocab_size: usize,
//     pub pad_token_id: u32,
// }

// struct XLMRobertaEmbeddings {
//     word_embeddings: Embedding,
//     position_embeddings: Option<Embedding>,
//     token_type_embeddings: Embedding,
//     layer_norm: LayerNorm,
//     padding_idx: u32,
//     span: tracing::Span,
// }

// impl XLMRobertaEmbeddings {
//     fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
//         let word_embeddings = embedding(
//             config.vocab_size,
//             config.hidden_size,
//             vb.pp("word_embeddings"),
//         )?;
//         let position_embeddings = embedding(
//             config.max_position_embeddings,
//             config.hidden_size,
//             vb.pp("position_embeddings"),
//         )?;
//         let token_type_embeddings = embedding(
//             config.type_vocab_size,
//             config.hidden_size,
//             vb.pp("token_type_embeddings"),
//         )?;
//         let layer_norm = layer_norm(
//             config.hidden_size,
//             config.layer_norm_eps,
//             vb.pp("LayerNorm"),
//         )?;
//         Ok(Self {
//             word_embeddings,
//             position_embeddings: Some(position_embeddings),
//             token_type_embeddings,
//             layer_norm,
//             padding_idx: config.pad_token_id,
//             span: tracing::span!(tracing::Level::TRACE, "embeddings"),
//         })
//     }

//     fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
//         let _enter = self.span.enter();
//         let (_bsize, _) = input_ids.dims2()?;
//         let input_embeddings = self.word_embeddings.forward(input_ids)?;
//         let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
//         let mut embeddings = (&input_embeddings + token_type_embeddings)?;
//         if let Some(position_embeddings) = &self.position_embeddings {
//             let mask = input_ids
//                 .ne(self.padding_idx)?
//                 .to_dtype(input_embeddings.dtype())?;
//             let cumsum = mask.cumsum(1)?;
//             let position_ids = (cumsum * mask)?
//                 .broadcast_add(
//                     &Tensor::try_from(self.padding_idx)?
//                         .to_dtype(input_embeddings.dtype())?
//                         .to_device(input_embeddings.device())?,
//                 )?
//                 .to_dtype(candle_core::DType::U32)?;
//             embeddings = embeddings.broadcast_add(&position_embeddings.forward(&position_ids)?)?;
//         }
//         let embeddings = self.layer_norm.forward(&embeddings)?;
//         Ok(embeddings)
//     }
// }

// struct XLMRobertaSelfAttention {
//     num_attention_heads: usize,
//     attention_head_size: usize,
//     all_head_size: usize,
//     query: Linear,
//     key: Linear,
//     value: Linear,
// }

// impl XLMRobertaSelfAttention {
//     fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
//         let attention_head_size = cfg.hidden_size / cfg.num_attention_heads;
//         let all_head_size = cfg.num_attention_heads * attention_head_size;
//         Ok(Self {
//             num_attention_heads: cfg.num_attention_heads,
//             attention_head_size,
//             all_head_size,
//             query: linear(cfg.hidden_size, all_head_size, vb.pp("query"))?,
//             key: linear(cfg.hidden_size, all_head_size, vb.pp("key"))?,
//             value: linear(cfg.hidden_size, all_head_size, vb.pp("value"))?,
//         })
//     }

//     fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
//         let mut new_x_shape = x.dims().to_vec();
//         new_x_shape[2] = self.num_attention_heads;
//         new_x_shape.push(self.attention_head_size);
//         let x = x.reshape(new_x_shape)?;
//         x.permute((0, 2, 1, 3))?.contiguous()
//     }

//     fn forward(
//         &self,
//         hidden_states: &Tensor,
//         encoder_hidden_states: Option<&Tensor>,
//         attention_mask: &Tensor,
//         past_key_value: Option<(&Tensor, &Tensor)>,
//         encoder_attention_mask: Option<&Tensor>,
//     ) -> Result<Tensor> {
//         let mixed_query_layer = self.query.forward(hidden_states)?;
//         let is_cross_attention = encoder_hidden_states.is_some();
//         let (key_layer, value_layer, attention_mask) = if is_cross_attention
//             && past_key_value.is_some()
//         {
//             let key_layer = past_key_value.unwrap().0.clone();
//             let value_layer = past_key_value.unwrap().1.clone();
//             let attention_mask = encoder_attention_mask.unwrap().clone();
//             (key_layer, value_layer, Some(attention_mask))
//         } else if is_cross_attention {
//             let key_layer =
//                 self.transpose_for_scores(&self.key.forward(encoder_hidden_states.unwrap())?)?;
//             let value_layer =
//                 self.transpose_for_scores(&self.value.forward(encoder_hidden_states.unwrap())?)?;
//             let attention_mask = encoder_attention_mask.unwrap();
//             (key_layer, value_layer, Some(attention_mask.clone()))
//         } else if past_key_value.is_some() {
//             let mut key_layer = self.transpose_for_scores(&self.key.forward(hidden_states)?)?;
//             let mut value_layer = self.transpose_for_scores(&self.value.forward(hidden_states)?)?;
//             key_layer = Tensor::cat(
//                 &[
//                     past_key_value.clone().as_ref().unwrap().0.clone(),
//                     key_layer,
//                 ],
//                 2,
//             )?;
//             value_layer = Tensor::cat(
//                 &[past_key_value.as_ref().unwrap().1.clone(), value_layer],
//                 2,
//             )?;
//             (key_layer, value_layer, Some(attention_mask.clone()))
//         } else {
//             let key_layer = self.transpose_for_scores(&self.key.forward(hidden_states)?)?;
//             let value_layer = self.transpose_for_scores(&self.value.forward(hidden_states)?)?;
//             (key_layer, value_layer, Some(attention_mask.clone()))
//         };

//         let query_layer = self.transpose_for_scores(&mixed_query_layer)?;
//         let mut attention_scores = query_layer.matmul(&key_layer.transpose(2, 3)?)?;
//         let scale = 1f64 / f64::sqrt(self.attention_head_size as f64);

//         attention_scores = (attention_scores * scale)?;
//         attention_scores = match attention_mask {
//             None => attention_scores,
//             Some(mask) => {
//                 attention_scores.broadcast_add(&mask.to_dtype(attention_scores.dtype())?)?
//             }
//         };
//         let attention_probs = softmax_last_dim(&attention_scores)?;

//         let context_layer = attention_probs
//             .matmul(&value_layer)?
//             .permute((0, 2, 1, 3))?
//             .contiguous()?;
//         let mut new_context_layer_shape =
//             context_layer.dims()[..context_layer.dims().len() - 2].to_vec();
//         new_context_layer_shape.push(self.all_head_size);
//         let context_layer = context_layer.reshape(new_context_layer_shape)?;

//         Ok(context_layer)
//     }
// }

// struct XLMRobertaSelfOutput {
//     dense: Linear,
//     layernorm: LayerNorm,
// }

// impl XLMRobertaSelfOutput {
//     fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
//         let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
//         let layernorm =
//             candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
//         Ok(Self { dense, layernorm })
//     }

//     fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
//         let hidden_states = self.dense.forward(hidden_states)?;
//         let hidden_states = self.layernorm.forward(&(hidden_states + input_tensor)?)?;
//         Ok(hidden_states)
//     }
// }

// struct XLMRobertaAttention {
//     output: XLMRobertaSelfOutput,
//     self_attention: XLMRobertaSelfAttention,
// }

// impl XLMRobertaAttention {
//     fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
//         let output = XLMRobertaSelfOutput::new(cfg, vb.pp("output"))?;
//         let self_attention = XLMRobertaSelfAttention::new(cfg, vb.pp("self"))?;
//         Ok(Self {
//             output,
//             self_attention,
//         })
//     }

//     fn forward(
//         &self,
//         hidden_states: &Tensor,
//         attention_mask: &Tensor,
//         encoder_hidden_states: Option<&Tensor>,
//         encoder_attention_mask: Option<&Tensor>,
//         past_key_value: Option<(&Tensor, &Tensor)>,
//     ) -> Result<(Tensor, Tensor)> {
//         let self_outputs = self.self_attention.forward(
//             hidden_states,
//             encoder_hidden_states,
//             attention_mask,
//             past_key_value,
//             encoder_attention_mask,
//         )?;
//         let attention_output = self.output.forward(&self_outputs, hidden_states)?;
//         Ok((attention_output, self_outputs))
//     }
// }

// struct XLMRobertaOutput {
//     dense: Linear,
//     layernorm: LayerNorm,
// }

// impl XLMRobertaOutput {
//     fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
//         let dense = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("dense"))?;
//         let layernorm =
//             candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
//         Ok(Self { dense, layernorm })
//     }

//     fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
//         let hidden_states = self.dense.forward(hidden_states)?;
//         let hidden_states = self.layernorm.forward(&(hidden_states + input_tensor)?)?;
//         Ok(hidden_states)
//     }
// }

// struct XLMRobertaIntermediate {
//     dense: Linear,
//     intermediate_act_fn: Activation,
// }

// impl XLMRobertaIntermediate {
//     fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
//         let dense = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("dense"))?;
//         let intermediate_act_fn = cfg.hidden_act;
//         Ok(Self {
//             dense,
//             intermediate_act_fn,
//         })
//     }

//     fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
//         let hidden_states = self.dense.forward(hidden_states)?;
//         let hidden_states = self.intermediate_act_fn.forward(&hidden_states)?;
//         Ok(hidden_states)
//     }
// }

// struct XLMRobertaLayer {
//     attention: XLMRobertaAttention,
//     intermediate: XLMRobertaIntermediate,
//     output: XLMRobertaOutput,
// }

// impl XLMRobertaLayer {
//     fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
//         let attention = XLMRobertaAttention::new(cfg, vb.pp("attention"))?;
//         let intermediate = XLMRobertaIntermediate::new(cfg, vb.pp("intermediate"))?;
//         let output = XLMRobertaOutput::new(cfg, vb.pp("output"))?;
//         Ok(Self {
//             attention,
//             intermediate,
//             output,
//         })
//     }

//     fn forward(
//         &self,
//         hidden_states: &Tensor,
//         attention_mask: &Tensor,
//         encoder_hidden_states: Option<&Tensor>,
//         encoder_attention_mask: Option<&Tensor>,
//         past_key_value: Option<(&Tensor, &Tensor)>,
//     ) -> Result<(Tensor, Tensor)> {
//         let self_attention_outputs = self.attention.forward(
//             hidden_states,
//             attention_mask,
//             encoder_hidden_states,
//             encoder_attention_mask,
//             past_key_value,
//         )?;
//         let attention_output = self_attention_outputs.0;
//         let outputs = self_attention_outputs.1;
//         let intermediate_output = self.intermediate.forward(&attention_output)?;
//         let layer_output = self
//             .output
//             .forward(&intermediate_output, &attention_output)?;
//         Ok((layer_output, outputs))
//     }
// }

// struct XLMRobertaEncoder {
//     layers: Vec<XLMRobertaLayer>,
// }

// impl XLMRobertaEncoder {
//     fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
//         let layers = (0..cfg.num_hidden_layers)
//             .map(|i| XLMRobertaLayer::new(cfg, vb.pp(format!("layer.{}", i))))
//             .collect::<Result<Vec<_>>>()?;
//         Ok(Self { layers })
//     }

//     fn forward(
//         &self,
//         hidden_states: &Tensor,
//         attention_mask: &Tensor,
//         encoder_hidden_states: Option<&Tensor>,
//         encoder_attention_mask: Option<&Tensor>,
//         past_key_value: Option<(&Tensor, &Tensor)>,
//     ) -> Result<Tensor> {
//         let mut hidden_states = hidden_states.clone();
//         for layer_module in self.layers.iter() {
//             let layer_outputs = layer_module.forward(
//                 &hidden_states,
//                 attention_mask,
//                 encoder_hidden_states,
//                 encoder_attention_mask,
//                 past_key_value,
//             )?;
//             hidden_states = layer_outputs.0;
//         }
//         Ok(hidden_states)
//     }
// }

// pub struct XLMRobertaModel {
//     encoder: XLMRobertaEncoder,
//     embeddings: XLMRobertaEmbeddings,
// }

// impl AutoModel for XLMRobertaModel {
//     type Config = Config;
//     type Model = Self;
//     fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
//         Self::new(config, vb)
//     }
// }

// impl XLMRobertaModel {
//     pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
//         let encoder = XLMRobertaEncoder::new(cfg, vb.pp("encoder"))?;
//         let embeddings = XLMRobertaEmbeddings::load(vb.pp("embeddings"), cfg)?;
//         Ok(Self {
//             encoder,
//             embeddings,
//         })
//     }

//     pub fn forward(
//         &self,
//         input_ids: &Tensor,
//         attention_mask: &Tensor,
//         token_type_ids: &Tensor,
//         past_key_value: Option<(&Tensor, &Tensor)>,
//         encoder_hidden_states: Option<&Tensor>,
//         encoder_attention_mask: Option<&Tensor>,
//     ) -> Result<Tensor> {
//         let hidden_states = self.embeddings.forward(input_ids, token_type_ids)?;
//         let attention_mask = prepare_4d_attention_mask(attention_mask, DType::F32, None)?
//             .to_device(hidden_states.device())?;
//         let hidden_states = self.encoder.forward(
//             &hidden_states,
//             &attention_mask,
//             encoder_hidden_states,
//             encoder_attention_mask,
//             past_key_value,
//         )?;
//         Ok(hidden_states)
//     }
// }

// struct XLMRobertaLMHead {
//     dense: Linear,
//     layer_norm: LayerNorm,
// }

// impl XLMRobertaLMHead {
//     fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
//         let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
//         let layer_norm =
//             candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm"))?;
//         Ok(Self { dense, layer_norm })
//     }

//     fn forward(&self, hidden_states: &Tensor, shared_embeddings: &Tensor) -> Result<Tensor> {
//         let hidden_states = self.dense.forward(hidden_states)?;
//         let hidden_states = candle_nn::Activation::Gelu.forward(&hidden_states)?;
//         let hidden_states = self.layer_norm.forward(&hidden_states)?;
//         let hidden_states = hidden_states.broadcast_matmul(shared_embeddings)?;
//         Ok(hidden_states)
//     }
// }

// pub struct XLMRobertaForMaskedLM {
//     roberta: XLMRobertaModel,
//     lm_head: XLMRobertaLMHead,
// }

// impl AutoModel for XLMRobertaForMaskedLM {
//     type Config = Config;
//     type Model = Self;
//     fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
//         Self::new(config, vb)
//     }
// }

// impl XLMRobertaForMaskedLM {
//     pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
//         let roberta = XLMRobertaModel::new(cfg, vb.pp("roberta"))?;
//         let lm_head = XLMRobertaLMHead::new(cfg, vb.pp("lm_head"))?;
//         Ok(Self { roberta, lm_head })
//     }

//     pub fn forward(
//         &self,
//         input_ids: &Tensor,
//         attention_mask: &Tensor,
//         token_type_ids: &Tensor,
//         past_key_value: Option<(&Tensor, &Tensor)>,
//         encoder_hidden_states: Option<&Tensor>,
//         encoder_attention_mask: Option<&Tensor>,
//     ) -> Result<Tensor> {
//         let hidden_states = self.roberta.forward(
//             input_ids,
//             attention_mask,
//             token_type_ids,
//             past_key_value,
//             encoder_hidden_states,
//             encoder_attention_mask,
//         )?;
//         let lm_logits = self.lm_head.forward(
//             &hidden_states,
//             &self
//                 .roberta
//                 .embeddings
//                 .word_embeddings
//                 .embeddings()
//                 .t()?
//                 .unsqueeze(0)?,
//         )?;
//         Ok(lm_logits)
//     }
// }

// struct XLMRobertaClassificationHead {
//     dense: Linear,
//     out_proj: Linear,
// }

// impl XLMRobertaClassificationHead {
//     fn new(num_labels: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
//         let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
//         let out_proj = linear(cfg.hidden_size, num_labels, vb.pp("out_proj"))?;
//         Ok(Self { dense, out_proj })
//     }

//     fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
//         let cls_states = hidden_states.get_on_dim(1, 0)?.contiguous()?;
//         let hidden_states = self.dense.forward(&cls_states)?;
//         let hidden_states = candle_nn::Activation::GeluPytorchTanh.forward(&hidden_states)?;
//         let hidden_states = self.out_proj.forward(&hidden_states)?;
//         Ok(hidden_states)
//     }
// }

// pub struct XLMRobertaForSequenceClassification {
//     roberta: XLMRobertaModel,
//     classifier: XLMRobertaClassificationHead,
// }

// impl XLMRobertaForSequenceClassification {
//     pub fn new(num_labels: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
//         let roberta = XLMRobertaModel::new(cfg, vb.pp("roberta"))?;
//         let classifier = XLMRobertaClassificationHead::new(num_labels, cfg, vb.pp("classifier"))?;
//         Ok(Self {
//             roberta,
//             classifier,
//         })
//     }

//     pub fn forward(
//         &self,
//         input_ids: &Tensor,
//         attention_mask: &Tensor,
//         token_type_ids: &Tensor,
//     ) -> Result<Tensor> {
//         let hidden_states =
//             self.roberta
//                 .forward(input_ids, attention_mask, token_type_ids, None, None, None)?;
//         self.classifier.forward(&hidden_states)
//     }
// }

// fn prepare_4d_attention_mask(
//     mask: &Tensor,
//     dtype: DType,
//     tgt_len: Option<usize>,
// ) -> Result<Tensor> {
//     let bsz = mask.dims()[0];
//     let src_len = mask.dims()[1];
//     let tgt_len = tgt_len.unwrap_or(src_len);

//     let expanded_mask = mask
//         .unsqueeze(1)?
//         .unsqueeze(2)?
//         .expand((bsz, 1, tgt_len, src_len))?
//         .to_dtype(dtype)?;

//     let inverted_mask = (1.0 - expanded_mask)?;

//     (inverted_mask * get_dtype_min_val(dtype))?.to_dtype(dtype)
// }

// fn get_dtype_min_val(dtype: DType) -> f64 {
//     match dtype {
//         DType::F32 => f32::MIN as f64,
//         DType::F64 => f64::MIN,
//         _ => panic!("Unsupported data type"),
//     }
// }
