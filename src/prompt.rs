use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub struct PromptStore {
    dir: PathBuf,
}

impl PromptStore {
    pub fn new(dir: &Path) -> Self {
        Self { dir: dir.to_path_buf() }
    }

    pub fn render(&self, file_name: &str, vars: &HashMap<String, String>) -> Result<String> {
        let path = self.dir.join(file_name);
        let mut template =
            fs::read_to_string(&path).with_context(|| format!("读取提示词失败: {:?}", path))?;

        for (k, v) in vars {
            template = template.replace(&format!("{{{{{}}}}}", k), v);
        }
        Ok(template)
    }
}