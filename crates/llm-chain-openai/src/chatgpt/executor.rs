use super::prompt::completion_to_output;
use super::prompt::stream_to_output;
use llm_chain::options::Opt;
use llm_chain::options::Options;
use llm_chain::options::OptionsCascade;
use llm_chain::output::Output;
use llm_chain::tokens::TokenCollection;

use super::prompt::create_chat_completion_request;
use super::prompt::format_chat_messages;
use async_openai::{error::OpenAIError,config::OpenAIConfig, types::ChatCompletionRequestMessage};
use llm_chain::prompt::Prompt;

use llm_chain::tokens::PromptTokensError;
use llm_chain::tokens::{Tokenizer, TokenizerError};
use llm_chain::traits;
use llm_chain::traits::{ExecutorCreationError, ExecutorError};

use async_trait::async_trait;
use llm_chain::tokens::TokenCount;

use tiktoken_rs::get_chat_completion_max_tokens;

use std::sync::Arc;
use tokio::time::{sleep, Duration};
/// The `Executor` struct for the ChatGPT model. This executor uses the `async_openai` crate to communicate with the OpenAI API.
#[derive(Clone)]
pub struct Executor {
    /// The client used to communicate with the OpenAI API.
    client: Arc<async_openai::Client<OpenAIConfig>>,
    /// The per-invocation options for this executor.
    options: Options,
}

impl Default for Executor {
    fn default() -> Self {
        let config = OpenAIConfig::default();
        let client = async_openai::Client::with_config(config);
        Self {
            client: Arc::new(client),
            options: Options::default()
        }
    }
}

impl Executor {
    /// Creates a new `Executor` with the given client.
    pub fn for_client(client: async_openai::Client<OpenAIConfig>, options: Options) -> Self {
        use llm_chain::traits::Executor as _;
        let mut exec = Self::new_with_options(options).unwrap();
        exec.client = Arc::new(client);
        exec
    }

    fn get_model_from_invocation_options(&self, opts: &OptionsCascade) -> String {
        let Some(Opt::Model(model)) = opts.get(llm_chain::options::OptDiscriminants::Model) else {
            return "gpt-3.5-turbo".to_string()
        };
        model.to_name()
    }

    fn cascade<'a>(&'a self, opts: Option<&'a Options>) -> OptionsCascade<'a> {
        let mut v: Vec<&'a Options> = vec![&self.options];
        if let Some(o) = opts {
            v.push(o);
        }
        OptionsCascade::from_vec(v)
    }
}

#[derive(thiserror::Error, Debug)]
#[error(transparent)]
pub enum Error {
    OpenAIError(#[from] OpenAIError),
}

#[async_trait]
impl traits::Executor for Executor {
    type StepTokenizer<'a> = OpenAITokenizer;
    /// Creates a new `Executor` with the given options.
    ///
    /// if the `OPENAI_ORG_ID` environment variable is present, it will be used as the org_ig for the OpenAI client.
    fn new_with_options(options: Options) -> Result<Self, ExecutorCreationError> {
        // let mut client = async_openai::Client::new();
        // let opts = OptionsCascade::new().with_options(&options);

        // if let Some(Opt::ApiKey(api_key)) = opts.get(llm_chain::options::OptDiscriminants::ApiKey) {
        //     client = client.with_api_key(api_key)
        // }

        // if let Ok(org_id) = std::env::var("OPENAI_ORG_ID") {
        //     client = client.with_org_id(org_id);
        // }
        
        let opts = OptionsCascade::new().with_options(&options);
        let open_ai_key = if let Some(Opt::ApiKey(api_key)) = opts.get(llm_chain::options::OptDiscriminants::ApiKey) {
            api_key
        } else {
            ""
        };

        let org_id = if let Ok(org_id) = std::env::var("OPENAI_ORG_ID") {
            org_id
        } else {
            "".to_string()
        };
        let config = OpenAIConfig::new().with_api_key(open_ai_key).with_org_id(org_id);
        let client = Arc::new(async_openai::Client::with_config(config));
        
        Ok(Self { client, options })
    }

    async fn execute(&self, options: &Options, prompt: &Prompt) -> Result<Output, ExecutorError> {
        let opts = self.cascade(Some(options));
        let client = self.client.clone();
        let model = self.get_model_from_invocation_options(&opts);
        let input = create_chat_completion_request(model, prompt, opts.is_streaming()).unwrap();
        let retry_client = client.clone();
        let retry_input = input.clone();
        if opts.is_streaming() {
            match async move { client.chat().create_stream(input).await }.await {
                Ok(client_normal) => {
                    Ok(stream_to_output(client_normal))
                }
                Err(err) => {
                    log::error!("llm-chain execute create_stream error = {},retry ", err);
                    sleep(Duration::from_millis(1000)).await;
                    let res = async move { retry_client.chat().create_stream(retry_input).await }
                    .await
                    .map_err(|e| ExecutorError::InnerError(e.into()))?;
                    Ok(stream_to_output(res))
                }
            }
        } else {
            match async move { client.chat().create(input).await }.await {
                Ok(client_normal) => {
                    match completion_to_output(client_normal) {
                        Ok(output) => {
                            Ok(output)
                        }
                        Err(err) => {
                             // retry one more time
                             log::error!("llm-chain execute competion error = {},retry", err);
                             let res = async move { retry_client.chat().create(retry_input).await }.await.map_err(|e| ExecutorError::InnerError(e.into()))?;
                             let output = completion_to_output(res).map_err(|err| ExecutorError::ResoponseCompleteError(err.to_string()))?;
                             Ok(output)
                         }
                    }
                }
                Err(err) => {
                    log::error!("llm-chain execute create error = {},retry ", err);
                    sleep(Duration::from_millis(1000)).await;
                    let res = async move { retry_client.chat().create(retry_input).await }
                    .await
                    .map_err(|e| ExecutorError::InnerError(e.into()))?;

                    let output = completion_to_output(res).map_err(|err| ExecutorError::ResoponseCompleteError(err.to_string()))?;
                    Ok(output)
                }
            }
        }
    }

    fn tokens_used(
        &self,
        opts: &Options,
        prompt: &Prompt,
    ) -> Result<TokenCount, PromptTokensError> {
        let opts_cas = self.cascade(Some(opts));
        let model = self.get_model_from_invocation_options(&opts_cas);
        let messages: Vec<ChatCompletionRequestMessage> = format_chat_messages(prompt.to_chat())?;
        let no_messages: Vec<tiktoken_rs::ChatCompletionRequestMessage> = Vec::new();
        let tokens_used = get_chat_completion_max_tokens(&model, no_messages.as_slice())
            .map_err(|_| PromptTokensError::NotAvailable)?
            - get_chat_completion_max_tokens(&model, as_tiktoken_messages(messages).as_slice())
                .map_err(|_| PromptTokensError::NotAvailable)?;

        Ok(TokenCount::new(
            self.max_tokens_allowed(opts),
            tokens_used as i32,
        ))
    }
    /// Get the context size from the model or return default context size
    fn max_tokens_allowed(&self, opts: &Options) -> i32 {
        let opts_cas = self.cascade(Some(opts));
        let model = self.get_model_from_invocation_options(&opts_cas);
        tiktoken_rs::model::get_context_size(&model)
            .try_into()
            .unwrap_or(4096)
    }

    fn answer_prefix(&self, _prompt: &Prompt) -> Option<String> {
        None
    }

    fn get_tokenizer(&self, options: &Options) -> Result<OpenAITokenizer, TokenizerError> {
        Ok(OpenAITokenizer::new(self.cascade(Some(options))))
    }
}

fn as_tiktoken_message(
    message: &ChatCompletionRequestMessage,
) -> tiktoken_rs::ChatCompletionRequestMessage {
    tiktoken_rs::ChatCompletionRequestMessage {
        role: message.role.to_string(),
        // content: Some(message.content.clone()),
        content: message.content.clone(),
        name: message.name.clone(),
        function_call: None
    }
}

fn as_tiktoken_messages(
    messages: Vec<ChatCompletionRequestMessage>,
) -> Vec<tiktoken_rs::ChatCompletionRequestMessage> {
    messages.iter().map(|x| as_tiktoken_message(x)).collect()
}

pub struct OpenAITokenizer {
    model_name: String,
}

impl OpenAITokenizer {
    pub fn new(options: OptionsCascade) -> Self {
        let model_name = match options.get(llm_chain::options::OptDiscriminants::Model) {
            Some(Opt::Model(model_name)) => model_name.to_name(),
            _ => "gpt-3.5-turbo".to_string(),
        };
        Self::for_model_name(model_name)
    }
    /// Creates an OpenAITokenizer for the passed in model name
    pub fn for_model_name<S: Into<String>>(model_name: S) -> Self {
        let model_name: String = model_name.into();
        Self { model_name }
    }

    fn get_bpe_from_model(&self) -> Result<tiktoken_rs::CoreBPE, PromptTokensError> {
        use tiktoken_rs::get_bpe_from_model;
        get_bpe_from_model(&self.model_name).map_err(|_| PromptTokensError::NotAvailable)
    }
}

// FIXME: unicode crash!
impl Tokenizer for OpenAITokenizer {
    // FIXME: not use!
    fn tokenize_str(&self, doc: &str) -> Result<TokenCollection, TokenizerError> {
        Ok(self
            .get_bpe_from_model()
            .map_err(|_| TokenizerError::TokenizationError)?
            .encode_ordinary(doc)
            .into())
    }

    // FIXME: not use!
    fn to_string(&self, tokens: TokenCollection) -> Result<String, TokenizerError> {
        let res = self
            .get_bpe_from_model()
            .map_err(|_e| TokenizerError::ToStringError)?
            .decode(tokens.as_usize()?)
            .map_err(|_e| TokenizerError::ToStringError)?;
        Ok(res)
    }
}
