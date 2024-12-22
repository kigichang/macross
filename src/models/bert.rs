use std::collections::HashMap;

use candle_core::{DType, D};
use candle_core::{Result, Tensor};
use candle_nn::Module;
use candle_nn::VarBuilder;
use serde::Deserialize;

use super::activations::HiddenAct;
use super::auto::AutoModel;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
    Relative,
    //Learned,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct BertConfig {
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
    #[serde(default)]
    pub position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    pub use_cache: bool,
    pub classifier_dropout: Option<f64>,
    pub model_type: Option<String>,
    pub id2label: Option<HashMap<String, String>>,
    pub label2id: Option<HashMap<String, usize>>,
}

impl BertConfig {
    pub fn new() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
            id2label: None,
            label2id: None,
        }
    }
}

pub struct BertEmbeddings {
    word_embeddings: crate::Embedding,
    position_embeddings: crate::Embedding,
    token_type_embeddings: crate::Embedding,
    layer_norm: crate::LayerNorm,
    dropout: crate::Dropout,
    // position_ids: Tensor,
    // token_type_ids: Tensor,
    position_embedding_type: PositionEmbeddingType,
}

impl BertEmbeddings {
    pub fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
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

        let mut embeddings = (&inputs_embeds + token_type_embeddings)?;
        if let PositionEmbeddingType::Absolute = self.position_embedding_type {
            let position_embeddings = self.position_embeddings.forward(&position_ids)?;
            embeddings = embeddings.broadcast_add(&position_embeddings)?;
        }
        let embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward(&embeddings)
    }
}

pub struct BertSelfAttention {
    num_attention_heads: usize,
    attention_head_size: usize,

    query: crate::Linear,
    key: crate::Linear,
    value: crate::Linear,
    dropout: crate::Dropout,
}

impl BertSelfAttention {
    pub fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;

        let query = crate::linear(config.hidden_size, all_head_size, vb.pp("query"))?;
        let key = crate::linear(config.hidden_size, all_head_size, vb.pp("key"))?;
        let value = crate::linear(config.hidden_size, all_head_size, vb.pp("value"))?;
        let dropout = crate::Dropout::new(config.attention_probs_dropout_prob);

        Ok(Self {
            num_attention_heads,
            attention_head_size,
            query,
            key,
            value,
            dropout,
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
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        context_layer.flatten_from(candle_core::D::Minus2)
    }
}

pub struct BertSelfOutput {
    dense: crate::Linear,
    layer_norm: crate::LayerNorm,
    dropout: crate::Dropout,
}

impl BertSelfOutput {
    pub fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let dense = crate::linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = crate::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = crate::Dropout::new(config.hidden_dropout_prob);

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
pub struct BertAttention {
    self_attention: BertSelfAttention,
    output: BertSelfOutput,
}

impl BertAttention {
    pub fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let self_attention = BertSelfAttention::new(vb.pp("self"), config)?;
        let output = BertSelfOutput::new(vb.pp("output"), config)?;

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

pub struct BertIntermediate {
    dense: crate::Linear,
    intermediate_act_fn: super::activations::HiddenActLayer,
}

impl BertIntermediate {
    pub fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let dense = crate::linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        let intermediate_act_fn = super::activations::HiddenActLayer::new(config.hidden_act);

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

pub struct BertOutput {
    dense: crate::Linear,
    layer_norm: crate::LayerNorm,
    dropout: crate::Dropout,
}

impl BertOutput {
    pub fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let dense = crate::linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = crate::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = crate::Dropout::new(config.hidden_dropout_prob);

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

pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    pub fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let attention = BertAttention::new(vb.pp("attention"), config)?;
        let intermediate = BertIntermediate::new(vb.pp("intermediate"), config)?;
        let output = BertOutput::new(vb.pp("output"), config)?;

        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let attention_outputs = self.attention.forward(hidden_states, attention_mask)?;
        self.feed_forward_chunk(&attention_outputs)
    }

    fn feed_forward_chunk(&self, attention_output: &Tensor) -> Result<Tensor> {
        let intermediate_output = self.intermediate.forward(attention_output)?;
        self.output.forward(&intermediate_output, attention_output)
    }
}

pub struct BertEncoder {
    layer: Vec<BertLayer>,
}

impl BertEncoder {
    pub fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let layer: Vec<BertLayer> = (0..config.num_hidden_layers)
            .map(|idx| BertLayer::new(vb.pp(format!("layer.{idx}")), config))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { layer })
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in self.layer.iter() {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }
}

pub struct BertPooler {
    dense: crate::Linear,
    //activation: fn(&Tensor) -> Tensor,
}

impl BertPooler {
    pub fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let dense = crate::linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;

        Ok(Self { dense })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // We "pool" the model by simply taking the hidden state corresponding to the first token.
        use candle_core::IndexOp;
        let first_token_tensor = hidden_states.i((.., 0))?.contiguous()?;
        let pooled_output = self.dense.forward(&first_token_tensor)?;
        pooled_output.tanh()
    }
}

pub struct BertPredictionHeadTransform {
    dense: crate::Linear,
    transform_act_fn: super::activations::HiddenActLayer,
    layer_norm: crate::LayerNorm,
}

impl BertPredictionHeadTransform {
    pub fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let dense = crate::linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let transform_act_fn = super::activations::HiddenActLayer::new(config.hidden_act);
        let layer_norm = crate::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;

        Ok(Self {
            dense,
            transform_act_fn,
            layer_norm,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(&hidden_states)?;
        let hidden_states = self.transform_act_fn.forward(&hidden_states)?;
        self.layer_norm.forward(&hidden_states)
    }
}

pub struct BertLMPredictionHead {
    transform: BertPredictionHeadTransform,
    decoder: crate::Linear,
}

impl BertLMPredictionHead {
    pub fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let transform = BertPredictionHeadTransform::new(vb.pp("transform"), config)?;
        let decoder = crate::linear(config.hidden_size, config.vocab_size, vb.pp("decoder"))?;

        Ok(BertLMPredictionHead { transform, decoder })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.transform.forward(hidden_states)?;
        self.decoder.forward(&hidden_states)
    }
}

pub struct BertOnlyMLMHead {
    predictions: BertLMPredictionHead,
}

impl BertOnlyMLMHead {
    pub fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let predictions = BertLMPredictionHead::new(vb.pp("predictions"), config)?;

        Ok(Self { predictions })
    }

    pub fn forward(&self, sequence_output: &Tensor) -> Result<Tensor> {
        self.predictions.forward(sequence_output)
    }
}

pub struct BertOnlyNSPHead {
    seq_relationship: crate::Linear,
}

impl BertOnlyNSPHead {
    pub fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let seq_relationship = crate::linear(config.hidden_size, 2, vb.pp("seq_relationship"))?;

        Ok(Self { seq_relationship })
    }

    pub fn forward(&self, pooled_output: &Tensor) -> Result<Tensor> {
        self.seq_relationship.forward(pooled_output)
    }
}

pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
}

impl AutoModel<BertConfig> for BertModel {
    type Model = Self;
    fn new(vb: VarBuilder, config: &BertConfig) -> Result<BertModel> {
        let embeddings = BertEmbeddings::new(vb.pp("embeddings"), config)?;
        let encoder = BertEncoder::new(vb.pp("encoder"), config)?;

        Ok(BertModel {
            embeddings,
            encoder,
        })
    }
}

impl BertModel {
    pub fn get_input_embeddings(&self) -> &crate::Embedding {
        &self.embeddings.word_embeddings
    }

    pub fn set_input_embeddings(&mut self, value: crate::Embedding) {
        self.embeddings.word_embeddings = value;
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let extended_attention_mask =
            self.get_extended_attention_mask(attention_mask, DType::F32)?;

        let embedding_output = self.embeddings.forward(&input_ids, &token_type_ids, None)?;
        self.encoder
            .forward(&embedding_output, &extended_attention_mask)
    }

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

pub struct BertModelWithPooler {
    bert: BertModel,
    pooler: BertPooler,
}

impl AutoModel<BertConfig> for BertModelWithPooler {
    type Model = Self;
    fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let bert = BertModel::new(vb.clone(), config)?;
        let pooler = BertPooler::new(vb.pp("pooler"), config)?;

        Ok(Self { bert, pooler })
    }
}

impl BertModelWithPooler {
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let encoder_outputs = self
            .bert
            .forward(input_ids, token_type_ids, attention_mask)?;
        self.pooler.forward(&encoder_outputs)
    }
}

pub struct BertLMHeadModel {
    bert: BertModel,
    cls: BertOnlyMLMHead,
}

impl BertLMHeadModel {
    pub fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let bert = BertModel::new(vb.pp("bert"), config)?;
        let cls = BertOnlyMLMHead::new(vb.pp("cls"), config)?;

        Ok(BertLMHeadModel { bert, cls })
    }

    pub fn get_output_embeddings(&self) -> &crate::Linear {
        &self.cls.predictions.decoder
    }

    pub fn set_output_embeddings(&mut self, new_embeddings: crate::Linear) {
        self.cls.predictions.decoder = new_embeddings;
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let sequence_output = self
            .bert
            .forward(&input_ids, &token_type_ids, &attention_mask)?;

        self.cls.forward(&sequence_output)
    }
}

pub struct BertForMaskedLM {
    bert: BertModel,
    cls: BertOnlyMLMHead,
}

impl AutoModel<BertConfig> for BertForMaskedLM {
    type Model = Self;
    fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let bert = BertModel::new(vb.pp("bert"), config)?;
        let cls = BertOnlyMLMHead::new(vb.pp("cls"), config)?;

        Ok(BertForMaskedLM { bert, cls })
    }
}

impl BertForMaskedLM {
    pub fn get_output_embeddings(&self) -> &crate::Linear {
        &self.cls.predictions.decoder
    }

    pub fn set_output_embeddings(&mut self, new_embeddings: crate::Linear) {
        self.cls.predictions.decoder = new_embeddings;
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let sequence_output = self
            .bert
            .forward(input_ids, token_type_ids, attention_mask)?;

        self.cls.forward(&sequence_output)
    }
}

pub struct BertForNextSentencePrediction {
    bert: BertModelWithPooler,
    cls: BertOnlyNSPHead,
}

impl BertForNextSentencePrediction {
    pub fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let bert = BertModelWithPooler::new(vb.pp("bert"), config)?;
        let cls = BertOnlyNSPHead::new(vb.pp("cls"), config)?;

        Ok(BertForNextSentencePrediction { bert, cls })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let pooled_output = self
            .bert
            .forward(input_ids, attention_mask, token_type_ids)?;

        self.cls.forward(&pooled_output)
    }
}

pub struct BertForSequenceClassification {
    bert: BertModelWithPooler,
    dropout: crate::Dropout,
    classifier: crate::Linear,
}

impl AutoModel<BertConfig> for BertForSequenceClassification {
    type Model = Self;
    fn new(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let bert = BertModelWithPooler::new(vb.pp("bert"), config)?;
        let classifier_dropout = config
            .classifier_dropout
            .unwrap_or(config.hidden_dropout_prob);
        let dropout = crate::Dropout::new(classifier_dropout);

        let num_labels = config.id2label.as_ref().map(|ids| ids.len()).unwrap_or(2);
        let classifier = crate::linear(config.hidden_size, num_labels, vb.pp("classifier"))?;

        Ok(BertForSequenceClassification {
            bert,
            dropout,
            classifier,
        })
    }
}

impl BertForSequenceClassification {
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let pooled_output = self
            .bert
            .forward(input_ids, token_type_ids, attention_mask)?;

        let pooled_output = self.dropout.forward(&pooled_output)?;
        self.classifier.forward(&pooled_output)
    }
}

pub fn mean_pooling(output: &Tensor, attention_mask: &Tensor) -> candle_core::Result<Tensor> {
    let attention_mask = attention_mask.unsqueeze(candle_core::D::Minus1)?;
    let input_mask_expanded = attention_mask
        .expand(output.shape())?
        .to_dtype(DType::F32)?;
    let sum = output.broadcast_mul(&input_mask_expanded)?.sum(1)?;
    let mask = input_mask_expanded.sum(1)?;
    let mask = mask.clamp(1e-9, f32::INFINITY)?;
    sum / mask
}
