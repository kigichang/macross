use candle_core::Tensor;

#[derive(Debug)]
pub struct SequenceClassifierOutput {
    pub loss: Option<Tensor>,
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
    pub attentions: Option<Tensor>,
}

#[derive(Debug)]
pub struct TokenClassifierOutput {
    pub loss: Option<Tensor>,
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
    pub attentions: Option<Tensor>,
}

#[derive(Debug)]
pub struct QuestionAnsweringModelOutput {
    pub loss: Option<Tensor>,
    pub start_logits: Tensor,
    pub end_logits: Tensor,
    pub hidden_states: Option<Tensor>,
    pub attentions: Option<Tensor>,
}
