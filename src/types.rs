use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleSegment {
    pub index: usize,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlobalContext {
    pub summary: String,
    #[serde(default)]
    pub glossary: Vec<GlossaryItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlossaryItem {
    pub source: String,
    pub target: String,
    #[serde(default)]
    pub note: Option<String>,
}

impl GlobalContext {
    pub fn glossary_as_text(&self) -> String {
        if self.glossary.is_empty() {
            return "（无）".to_string();
        }
        self.glossary
            .iter()
            .map(|g| {
                format!(
                    "- {} -> {}{}",
                    g.source,
                    g.target,
                    g.note
                        .as_ref()
                        .map(|n| format!("（{}）", n))
                        .unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}