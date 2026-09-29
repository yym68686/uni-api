use crate::config::snapshot::Provider;
use serde_json::Value;
use std::sync::Arc;

// Preserve the gateway's existing safety ceiling, including the first request.
const MAX_ATTEMPTS: usize = 100;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum RetryPolicy {
    #[default]
    Automatic,
    Retries(usize),
}

impl RetryPolicy {
    pub(crate) fn from_value(value: Option<&Value>) -> Self {
        match value {
            None | Some(Value::Null) | Some(Value::Bool(true)) => Self::Automatic,
            Some(Value::Number(number)) => Self::from_number(number.as_u64()),
            Some(Value::String(text)) => {
                let text = text.trim();
                if let Ok(number) = text.parse::<u64>() {
                    return Self::from_number(Some(number));
                }
                match text.to_ascii_lowercase().as_str() {
                    "true" | "t" | "yes" | "y" | "on" => Self::Automatic,
                    // Invalid values must not silently enable automatic retries.
                    _ => Self::Retries(0),
                }
            }
            _ => Self::Retries(0),
        }
    }

    fn from_number(number: Option<u64>) -> Self {
        Self::Retries(number.unwrap_or(0).min((MAX_ATTEMPTS - 1) as u64) as usize)
    }

    pub(crate) fn enabled(self) -> bool {
        self != Self::Retries(0)
    }

    pub(crate) fn max_attempts(self, providers: &[Arc<Provider>], targeted: bool) -> usize {
        if targeted {
            return 1;
        }
        match self {
            Self::Retries(retries) => retries.saturating_add(1).min(MAX_ATTEMPTS),
            Self::Automatic => {
                let extra = if providers.len() == 1 && providers[0].api_keys.len() > 1 {
                    providers[0].api_keys.len()
                } else {
                    providers
                        .iter()
                        .fold(0usize, |total, provider| {
                            total.saturating_add(provider.api_keys.len())
                        })
                        .saturating_mul(2)
                        .min(10)
                };
                providers
                    .len()
                    .saturating_add(extra.max(1))
                    .min(MAX_ATTEMPTS)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn retry_values_have_one_interpretation() {
        assert_eq!(RetryPolicy::from_value(None), RetryPolicy::Automatic);
        for value in [json!(true), Value::Null, json!(" TRUE "), json!("on")] {
            assert_eq!(
                RetryPolicy::from_value(Some(&value)),
                RetryPolicy::Automatic
            );
        }
        for value in [
            json!(false),
            json!(0),
            json!("false"),
            json!(" OFF "),
            json!("0"),
            json!(-1),
            json!(1.5),
            json!("invalid"),
            json!([]),
            json!({}),
        ] {
            assert_eq!(
                RetryPolicy::from_value(Some(&value)),
                RetryPolicy::Retries(0)
            );
        }
        for value in [json!(3), json!(" 3 ")] {
            let policy = RetryPolicy::from_value(Some(&value));
            assert_eq!(policy, RetryPolicy::Retries(3));
            assert_eq!(policy.max_attempts(&[], false), 4);
            assert_eq!(policy.max_attempts(&[], true), 1);
        }
        for value in [json!(u64::MAX), json!(u64::MAX.to_string())] {
            assert_eq!(
                RetryPolicy::from_value(Some(&value)).max_attempts(&[], false),
                100
            );
        }
    }
}
