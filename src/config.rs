use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub api_key: String,
    pub base_url: String,
    pub protocol: String,
    pub model: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.openai.com/v1".to_string(),
            protocol: "open_chat".to_string(),
            model: "gpt-4".to_string(),
        }
    }
}

fn config_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".ax")
}

fn config_path() -> PathBuf {
    config_dir().join("settings.json")
}

impl Config {
    pub fn load() -> Self {
        let path = config_path();
        if path.exists() {
            let data = fs::read_to_string(&path).unwrap_or_default();
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let dir = config_dir();
        fs::create_dir_all(&dir).map_err(|e| format!("Failed to create config dir: {e}"))?;
        let path = config_path();
        let data = serde_json::to_string_pretty(self).map_err(|e| format!("Failed to serialize: {e}"))?;
        fs::write(&path, data).map_err(|e| format!("Failed to write config: {e}"))?;
        Ok(())
    }

    pub fn effective_base_url(&self) -> String {
        if self.base_url != Config::default().base_url {
            return self.base_url.clone();
        }
        match self.protocol.as_str() {
            "anthropic_message" => "https://api.anthropic.com".to_string(),
            _ => "https://api.openai.com/v1".to_string(),
        }
    }

    pub fn effective_model(&self) -> String {
        if self.model != Config::default().model {
            return self.model.clone();
        }
        match self.protocol.as_str() {
            "anthropic_message" => "claude-sonnet-4-20250514".to_string(),
            _ => "gpt-4".to_string(),
        }
    }
}

pub fn apply_setting(key: &str, value: &str) -> Result<String, String> {
    let mut config = Config::load();
    let key_upper = key.to_uppercase();
    match key_upper.as_str() {
        "API_KEY" => {
            config.api_key = value.to_string();
            config.save()?;
            Ok(format!("API_KEY saved."))
        }
        "BASE_URL" => {
            config.base_url = value.to_string();
            config.save()?;
            Ok(format!("BASE_URL saved: {value}"))
        }
        "PROTOCOL" => {
            let valid = ["open_chat", "openai_response", "anthropic_message"];
            if !valid.contains(&value) {
                return Err(format!("Invalid protocol '{value}'. Valid: {}", valid.join(", ")));
            }
            config.protocol = value.to_string();
            config.save()?;
            Ok(format!("PROTOCOL saved: {value}"))
        }
        "MODEL" => {
            config.model = value.to_string();
            config.save()?;
            Ok(format!("MODEL saved: {value}"))
        }
        _ => Err(format!("Unknown setting: {key}. Valid: API_KEY, BASE_URL, PROTOCOL, MODEL")),
    }
}
