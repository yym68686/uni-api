use serde_json::Value;

pub(crate) fn content_text(value: Option<&Value>) -> Option<String> {
    let value = value?;
    if let Some(text) = value
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        return Some(text.to_owned());
    }
    let parts = value.as_array()?;
    let mut output = String::new();
    for part in parts {
        let text = part
            .get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty());
        if let Some(text) = text {
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(text);
        }
    }
    (!output.is_empty()).then_some(output)
}
