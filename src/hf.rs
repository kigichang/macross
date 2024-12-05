use hf_hub::{api::sync::Api, Repo, RepoType};

#[derive(Debug, Clone)]
pub struct ModelRepo {
    pub repo_id: String,
    pub revision: String,
}

impl Default for ModelRepo {
    fn default() -> Self {
        Self {
            repo_id: "".to_owned(),
            revision: "main".to_owned(),
        }
    }
}

impl ModelRepo {
    pub fn with_revision(repo_id: String, revision: String) -> Self {
        Self { repo_id, revision }
    }

    pub(crate) fn download(&self, file: &str) -> anyhow::Result<std::path::PathBuf> {
        let repo =
            Repo::with_revision(self.repo_id.clone(), RepoType::Model, self.revision.clone());
        let api = Api::new()?.repo(repo);
        Ok(api.get(file)?)
    }
}

impl From<(&str, &str)> for ModelRepo {
    fn from((repo_id, revision): (&str, &str)) -> Self {
        Self {
            repo_id: repo_id.to_owned(),
            revision: revision.to_owned(),
        }
    }
}

impl From<&str> for ModelRepo {
    fn from(repo_id: &str) -> Self {
        Self {
            repo_id: repo_id.to_owned(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct PretrainedModel {
    pub repo: ModelRepo,
    pub config: String,
    pub model: String,
}

impl Default for PretrainedModel {
    fn default() -> Self {
        Self {
            repo: Default::default(),
            config: "config.json".to_string(),
            model: "model.safetensors".to_string(),
        }
    }
}

impl PretrainedModel {
    pub fn new(repo_id: &str) -> Self {
        (repo_id, "main").into()
    }

    pub fn new_pth(repo_id: &str) -> Self {
        (repo_id, true).into()
    }

    pub fn with_files(repo_id: &str, config_file: &str, model_file: &str) -> Self {
        Self {
            repo: repo_id.into(),
            config: config_file.to_string(),
            model: model_file.to_string(),
        }
    }

    pub fn config(&self) -> anyhow::Result<std::path::PathBuf> {
        self.repo.download(&self.config)
    }

    pub fn model(&self) -> anyhow::Result<std::path::PathBuf> {
        self.repo.download(&self.model)
    }
}

impl From<(&str, &str, bool)> for PretrainedModel {
    fn from((repo_id, revision, pth): (&str, &str, bool)) -> Self {
        Self {
            repo: (repo_id, revision).into(),
            model: if pth {
                "pytorch_model.bin".to_string()
            } else {
                "model.safetensors".to_string()
            },
            ..Default::default()
        }
    }
}

impl From<(&str, &str)> for PretrainedModel {
    fn from((repo_id, revision): (&str, &str)) -> Self {
        (repo_id, revision, false).into()
    }
}

impl From<&str> for PretrainedModel {
    fn from(repo_id: &str) -> Self {
        (repo_id, "main").into()
    }
}

impl From<(&str, bool)> for PretrainedModel {
    fn from((repo_id, pth): (&str, bool)) -> Self {
        (repo_id, "main", pth).into()
    }
}
