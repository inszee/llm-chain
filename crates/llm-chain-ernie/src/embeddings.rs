
use std::sync::Arc;
use erniebot_rs::embedding::{EmbeddingEndpoint, EmbeddingModel};
use erniebot_rs::errors::ErnieError;
use async_trait::async_trait;
use llm_chain::traits::{self, EmbeddingsError};
use thiserror::Error;

pub struct Embeddings {
    client: Arc<EmbeddingEndpoint>
}

#[derive(Debug, Error)]
#[error(transparent)]
pub enum ErnieEmbeddingsError {
    #[error(transparent)]
    Client(#[from] ErnieError),
    #[error("Request to Ernie embeddings API was successful but response is empty")]
    EmptyResponse,
}

impl EmbeddingsError for ErnieEmbeddingsError {}

#[async_trait]
impl traits::Embeddings for Embeddings {
    type Error = ErnieEmbeddingsError;

    async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, Self::Error> {
        self.client.ainvoke(&texts, None).await
        .map(|r| 
            r.get_embedding_results().unwrap_or(vec![]).iter().map(|item| item.iter().map(|f| *f as f32).collect()).collect())
        .map_err(|e| e.into())
    }

    async fn embed_query(&self, query: String) -> Result<Vec<f32>, Self::Error> {
        let texts = vec![query];
        let results:Result<Vec<Vec<f32>>, Self::Error>  = self.client.ainvoke(&texts, None).await
        .map(|r| 
            r.get_embedding_results().unwrap_or(vec![]).iter().map(|item| item.iter().map(|f| *f as f32).collect()).collect())
        .map_err(|e| e.into());
        match results {
            Ok(vecs) => {
                if vecs.len() > 0 {
                    Ok(vecs.get(0).unwrap().to_vec())
                } else {
                    Ok(vec![])
                }
            }
            Err(err) => {
                std::result::Result::Err(err)
            }
        }
    }
}

impl Default for Embeddings {
    fn default() -> Self {
        let client = Arc::new(EmbeddingEndpoint::new(EmbeddingModel::EmbeddingV1).unwrap());
        Self {
            client
        }
    }
}

impl Embeddings {
    pub fn for_client(client: EmbeddingEndpoint) -> Self {
        Self {
            client: client.into()
        }
    }
}
