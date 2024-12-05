use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo};
use tokenizers::Tokenizer;

pub mod activations;
pub mod models;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LoadHFModel {
    pub model: String,
    pub config: String,
    pub tokenizer: String,
    pub pth: bool,
}

impl Default for LoadHFModel {
    fn default() -> Self {
        Self {
            model: "model.safetensors".to_string(),
            config: "config.json".to_string(),
            tokenizer: "tokenizer.json".to_string(),
            pth: false,
        }
    }
}

impl LoadHFModel {
    pub fn load_tokenizer(&self) -> tokenizers::Result<Tokenizer> {
        Tokenizer::from_file(&self.tokenizer)
    }

    pub fn load_model(&self, dtype: DType, device: &Device) -> candle_core::Result<VarBuilder> {
        if self.pth {
            VarBuilder::from_pth(&self.model, dtype, device)
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[&self.model], dtype, device) }
        }
    }

    pub fn load_config<T: serde::de::DeserializeOwned>(&self) -> anyhow::Result<T> {
        let reader = std::fs::File::open(&self.config)?;
        serde_json::from_reader(reader).map_err(anyhow::Error::msg)
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct HFModel {
    pub repo_id: String,
    pub revision: String,
    pub config: String,
    pub model: String,
    pub tokenizer: String,
}

impl Default for HFModel {
    fn default() -> Self {
        Self {
            repo_id: "".to_owned(),
            revision: "main".to_owned(),
            config: "config.json".to_string(),
            model: "model.safetensors".to_string(),
            tokenizer: "tokenizer.json".to_string(),
        }
    }
}

impl HFModel {
    pub fn new(repo_id: &str) -> Self {
        Self {
            repo_id: repo_id.to_owned(),
            ..Default::default()
        }
    }
    pub fn new_pth(repo_id: &str) -> Self {
        Self {
            repo_id: repo_id.to_owned(),
            model: "pytorch_model.bin".to_owned(),
            ..Default::default()
        }
    }

    pub fn get_ignore_tokenizer(&self, pth: bool) -> anyhow::Result<LoadHFModel> {
        let repo = Repo::with_revision(
            self.repo_id.clone(),
            hf_hub::RepoType::Model,
            self.revision.clone(),
        );
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get(&self.config)?.to_string_lossy().to_string();
        let model = api.get(&self.model)?.to_string_lossy().to_string();
        Ok(LoadHFModel {
            model,
            config,
            tokenizer: self.tokenizer.clone(),
            pth,
        })
    }

    pub fn get(&self, pth: bool) -> anyhow::Result<LoadHFModel> {
        let repo = Repo::with_revision(
            self.repo_id.clone(),
            hf_hub::RepoType::Model,
            self.revision.clone(),
        );
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get(&self.config)?.to_string_lossy().to_string();
        let tokenizer = api.get(&self.tokenizer)?.to_string_lossy().to_string();
        let model = api.get(&self.model)?.to_string_lossy().to_string();
        Ok(LoadHFModel {
            model,
            config,
            tokenizer,
            pth,
        })
    }
}

// from: https://github.com/huggingface/candle/blob/main/candle-examples/src/lib.rs
pub fn device(cpu: bool) -> candle_core::Result<Device> {
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
