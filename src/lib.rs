use candle_core::{Device, Result, Tensor};

pub mod hf;
pub mod models;

use candle_nn::{Module, VarBuilder};
pub use hf::*;
pub use models::auto::*;
use tracing::span;

#[derive(Debug, Clone)]
pub struct Dropout {
    #[allow(dead_code)]
    pr: f64,
}

impl Dropout {
    pub fn new(pr: f64) -> Self {
        Self { pr }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Embedding {
    inner: candle_nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    pub fn new(in_size: usize, out_size: usize, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::embedding(in_size, out_size, vb)?;
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn from_weights(weights: Tensor) -> Result<Self> {
        let (_in_size, out_size) = weights.dims2()?;
        let inner = candle_nn::Embedding::new(weights, out_size);
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn embeddings(&self) -> &Tensor {
        &self.inner.embeddings()
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(input)
    }
}

pub fn embedding(in_size: usize, out_size: usize, vb: VarBuilder) -> Result<Embedding> {
    Embedding::new(in_size, out_size, vb)
}

#[derive(Debug, Clone)]
pub struct Linear {
    inner: candle_nn::Linear,
    span: tracing::Span,
}

impl Linear {
    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> Self {
        let inner = candle_nn::Linear::new(weight, bias);
        let span = tracing::span!(tracing::Level::TRACE, "linear");
        Self { inner, span }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(input)
    }
}

pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear(in_dim, out_dim, vb)?;
    let span = span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

pub fn linear_b(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear_b(in_dim, out_dim, bias, vb)?;
    let span = span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear_no_bias(in_dim, out_dim, vb)?;
    let span = span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

#[derive(Debug, Clone)]
pub struct LayerNorm {
    inner: candle_nn::LayerNorm,
    span: tracing::Span,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        let inner = candle_nn::LayerNorm::new(weight, bias, eps);
        let span = tracing::span!(tracing::Level::TRACE, "layer_norm");
        Self { inner, span }
    }

    pub fn new_no_bias(weight: Tensor, eps: f64) -> Self {
        let inner = candle_nn::LayerNorm::new_no_bias(weight, eps);
        let span = tracing::span!(tracing::Level::TRACE, "layer_norm");
        Self { inner, span }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(input)
    }
}

pub fn layer_norm<C: Into<candle_nn::LayerNormConfig>>(
    size: usize,
    config: C,
    vb: VarBuilder,
) -> Result<LayerNorm> {
    let inner = candle_nn::layer_norm(size, config, vb)?;
    let span = span!(tracing::Level::TRACE, "layer_norm");
    Ok(LayerNorm { inner, span })
}

// from: https://github.com/huggingface/candle/blob/main/candle-examples/src/lib.rs
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle_core::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle_core::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    inner: candle_nn::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        let inner = candle_nn::RmsNorm::new(weight, eps);
        let span = tracing::span!(tracing::Level::TRACE, "rms_norm");
        Self { inner, span }
    }

    pub fn forward_diff(&self, input: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward_diff(input)
    }
}

impl Module for RmsNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(input)
    }
}

pub fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let inner = candle_nn::rms_norm(size, eps, vb)?;
    let span = span!(tracing::Level::TRACE, "rms_norm");
    Ok(RmsNorm { inner, span })
}

pub fn normalize(t: &Tensor) -> Result<Tensor> {
    let length = t.sqr()?.sum_keepdim(candle_core::D::Minus1)?.sqrt()?;
    t.broadcast_div(&length)
}
