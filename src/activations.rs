use candle_core::Result;
use candle_core::Tensor;
use candle_nn::ops::{sigmoid, swiglu};
use serde::Deserialize;

// https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/activations.py#L213
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    GeluApproximate,
    Relu,
    #[serde(alias = "silu")]
    Swiglu,
    Tanh,
    Sigmoid,
}

pub(crate) struct HiddenActLayer {
    act: HiddenAct,
    span: tracing::Span,
}

impl HiddenActLayer {
    pub fn new(act: HiddenAct) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "hidden-act");
        Self { act, span }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        match self.act {
            HiddenAct::Gelu => xs.gelu_erf(),
            HiddenAct::GeluApproximate => xs.gelu(),
            HiddenAct::Relu => xs.relu(),
            HiddenAct::Swiglu => swiglu(xs),
            HiddenAct::Tanh => xs.tanh(),
            HiddenAct::Sigmoid => sigmoid(xs),
        }
    }
}
