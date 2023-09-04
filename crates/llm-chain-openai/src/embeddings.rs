use std::sync::Arc;

use async_openai::{
    error::OpenAIError,
    config::{OpenAIConfig},
    types::{CreateEmbeddingRequest, EmbeddingInput},
};
use async_trait::async_trait;
use llm_chain::traits::{self, EmbeddingsError};
use thiserror::Error;

pub struct Embeddings {
    client: Arc<async_openai::Client<OpenAIConfig>>,
    model: String,
}

#[derive(Debug, Error)]
#[error(transparent)]
pub enum OpenAIEmbeddingsError {
    #[error(transparent)]
    Client(#[from] OpenAIError),
    #[error("Request to OpenAI embeddings API was successful but response is empty")]
    EmptyResponse,
}

impl EmbeddingsError for OpenAIEmbeddingsError {}

#[async_trait]
impl traits::Embeddings for Embeddings {
    type Error = OpenAIEmbeddingsError;

    async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, Self::Error> {
        self.client
            .embeddings()
            .create(CreateEmbeddingRequest {
                model: self.model.clone(),
                user: None,
                input: EmbeddingInput::from(texts),
            })
            .await
            .map(|r| r.data.into_iter().map(|e| e.embedding).collect())
            .map_err(|e| e.into())
    }

    async fn embed_query(&self, query: String) -> Result<Vec<f32>, Self::Error> {
        self.client
            .embeddings()
            .create(CreateEmbeddingRequest {
                model: self.model.clone(),
                user: None,
                input: EmbeddingInput::from(query),
            })
            .await
            .map(|r| r.data.into_iter())?
            .map(|e| e.embedding)
            .last()
            .ok_or(OpenAIEmbeddingsError::EmptyResponse)
    }
}

impl Default for Embeddings {
    fn default() -> Self {
        let config = OpenAIConfig::default();
        let client = async_openai::Client::with_config(config);
        Self {
            client: Arc::new(client),
            model: "text-embedding-ada-002".to_string(),
        }
    }
}

impl Embeddings {
    pub fn for_client(client: async_openai::Client<OpenAIConfig>, model: &str) -> Self {
        Self {
            client: client.into(),
            model: model.to_string(),
        }
    }
}
