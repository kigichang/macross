use crate::{ModelRepo, PretrainedModel};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::{
    ops::{Deref, DerefMut},
    path::Path,
};
use tokenizers::Tokenizer;

fn is_safetenors<P: AsRef<Path>>(model_file: P) -> bool {
    if let Some(ext) = model_file.as_ref().extension() {
        ext == "safetensors"
    } else {
        false
    }
}
pub struct AutoTokenizer(Tokenizer);

impl Deref for AutoTokenizer {
    type Target = Tokenizer;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for AutoTokenizer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AutoTokenizer {
    pub fn from_local<P: AsRef<Path>>(tokenizer_file: P) -> tokenizers::Result<Self> {
        Ok(Self(Tokenizer::from_file(tokenizer_file)?))
    }

    pub fn from_pretrained(model_repo: ModelRepo) -> tokenizers::Result<Self> {
        let tokenizer_file = model_repo.download("tokenizer.json")?;
        Ok(Self(Tokenizer::from_file(tokenizer_file)?))
    }

    pub fn from_pretrained_with_revision(
        repo_id: &str,
        revision: &str,
    ) -> tokenizers::Result<Self> {
        Self::from_pretrained((repo_id, revision).into())
    }
}

pub trait AutoModel<C: serde::de::DeserializeOwned> {
    type Model;
    fn from_local<P: AsRef<Path>>(
        config_file: P,
        model_file: P,
        dtype: DType,
        device: &Device,
    ) -> anyhow::Result<Self::Model> {
        let reader = std::fs::File::open(config_file)?;
        let config: C = serde_json::from_reader(reader)?;

        let vb = if is_safetenors(&model_file) {
            println!("loading from safetensors");
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, device)? }
        } else {
            VarBuilder::from_pth(model_file, dtype, device)?
        };

        Self::auto_load(vb, &config).map_err(anyhow::Error::msg)
    }

    fn from_pretrained(
        pretrained_model: PretrainedModel,
        dtype: DType,
        device: &Device,
    ) -> anyhow::Result<Self::Model> {
        let config = pretrained_model.config()?;
        let model = pretrained_model.model()?;
        Self::from_local(config, model, dtype, device)
    }

    fn auto_load(vb: VarBuilder, config: &C) -> candle_core::Result<Self::Model>;
}
