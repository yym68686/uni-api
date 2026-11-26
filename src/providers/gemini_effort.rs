//! Gemini 4 effort is a named level, not a legacy token budget.
use crate::config::snapshot::Provider;
use serde_json::{json, Map, Value};
use url::Url;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Preset {
    Standard,
    Extended,
}

impl Preset {
    fn for_model(model: &str) -> Option<Self> {
        match model.rsplit('/').next()? {
            "gemini-4-neon" | "gemini-4-hydrogen" => Some(Self::Extended),
            "gemini-4-carbon" | "gemini-4-argon" | "gemini-4-argon-a" => Some(Self::Standard),
            _ => None,
        }
    }

    pub(crate) fn intersect(self, other: Self) -> Self {
        if self == Self::Extended && other == Self::Extended {
            Self::Extended
        } else {
            Self::Standard
        }
    }

    pub(crate) fn default_level(self) -> &'static str {
        match self {
            Self::Standard => "medium",
            Self::Extended => "high",
        }
    }

    fn levels(self) -> &'static [(&'static str, &'static str)] {
        const LEVELS: [(&str, &str); 4] = [
            ("low", "Faster responses with lighter reasoning"),
            ("medium", "Balances speed and reasoning depth"),
            ("high", "Deeper reasoning for complex tasks"),
            ("xhigh", "Extra high reasoning for the hardest tasks"),
        ];
        match self {
            Self::Standard => &LEVELS[..3],
            Self::Extended => &LEVELS,
        }
    }

    pub(crate) fn options(self) -> Value {
        json!(self
            .levels()
            .iter()
            .map(|(effort, description)| json!({"effort":effort,"description":description}))
            .collect::<Vec<_>>())
    }

    pub(crate) fn level(self, effort: &str) -> Option<&'static str> {
        let effort = if effort.eq_ignore_ascii_case("extra_high")
            || effort.eq_ignore_ascii_case("extra-high")
        {
            "xhigh"
        } else {
            effort
        };
        self.levels()
            .iter()
            .find(|(level, _)| level.eq_ignore_ascii_case(effort))
            .map(|(level, _)| *level)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Wire {
    Gemini,
    Responses,
    Chat,
    OpenRouterChat,
    OpenRouterMessages,
}

pub(crate) fn route(provider: &Provider, upstream_model: &str) -> Option<(Preset, Wire)> {
    let preset = Preset::for_model(upstream_model)?;
    let openrouter = || {
        Url::parse(&provider.base_url)
            .is_ok_and(|url| crate::config::model_protocol::openrouter_api(&url))
    };
    let wire = match provider.engine.as_ref() {
        "gemini" | "vertex" | "vertex-gemini" => Wire::Gemini,
        "codex" => Wire::Responses,
        "gpt" | "openrouter"
            if provider
                .base_url
                .to_ascii_lowercase()
                .contains("/responses") =>
        {
            Wire::Responses
        }
        "gpt" | "openrouter" if openrouter() => Wire::OpenRouterChat,
        "gpt" | "openrouter" => Wire::Chat,
        // Only this Messages gateway translates effort for Google models.
        "claude" if upstream_model.starts_with("google/") && openrouter() => {
            Wire::OpenRouterMessages
        }
        _ => return None,
    };
    Some((preset, wire))
}

pub(crate) fn selected(
    preset: Preset,
    upstream_model: &str,
    payload: &Value,
) -> Result<Option<&'static str>, String> {
    let nested = payload.pointer("/reasoning/effort").and_then(Value::as_str);
    let flat = payload.get("reasoning_effort").and_then(Value::as_str);
    let Some(effort) = flat
        .or(nested)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    // Older Codex catalogs omitted the default, which some clients encode as
    // "none". Keep those requests on the upstream default.
    if effort.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    preset
        .level(effort)
        .map(Some)
        .ok_or_else(|| format!("{upstream_model} does not support reasoning effort {effort}"))
}

fn object_at<'a>(parent: &'a mut Map<String, Value>, key: &str) -> &'a mut Map<String, Value> {
    let child = parent.entry(key).or_insert_with(|| json!({}));
    if !child.is_object() {
        *child = json!({});
    }
    child.as_object_mut().expect("object ensured above")
}

pub(crate) fn remove_openai_effort(root: &mut Map<String, Value>) {
    root.remove("reasoning_effort");
    if let Some(reasoning) = root.get_mut("reasoning").and_then(Value::as_object_mut) {
        reasoning.remove("effort");
        if reasoning.is_empty() {
            root.remove("reasoning");
        }
    }
}

impl Wire {
    pub(crate) fn apply(self, root: &mut Map<String, Value>, effort: &str) {
        match self {
            Self::Gemini => {
                // Preserve generation options and thought visibility, but emit
                // only one level and never pair it with a legacy budget.
                let generation = object_at(root, "generationConfig");
                let thinking = object_at(generation, "thinkingConfig");
                for key in ["thinkingBudget", "thinking_budget", "thinking_level"] {
                    thinking.remove(key);
                }
                let level = match effort {
                    "low" => "LOW",
                    "medium" => "MEDIUM",
                    "high" => "HIGH",
                    "xhigh" => "XHIGH",
                    _ => unreachable!("validated Gemini level"),
                };
                thinking.insert("thinkingLevel".into(), json!(level));
                if let Some(config) = generation
                    .get_mut("thinking_config")
                    .and_then(Value::as_object_mut)
                {
                    for key in [
                        "thinkingBudget",
                        "thinking_budget",
                        "thinkingLevel",
                        "thinking_level",
                    ] {
                        config.remove(key);
                    }
                }
            }
            Self::Responses | Self::OpenRouterChat => {
                root.remove("reasoning_effort");
                object_at(root, "reasoning").insert("effort".into(), json!(effort));
            }
            Self::Chat => {
                remove_openai_effort(root);
                root.insert("reasoning_effort".into(), json!(effort));
            }
            Self::OpenRouterMessages => {
                let thinking = object_at(root, "thinking");
                thinking.remove("budget_tokens");
                thinking.insert("type".into(), json!("adaptive"));
                object_at(root, "output_config").insert("effort".into(), json!(effort));
            }
        }
    }
}
