//! 提示词模板渲染。
//!
//! 旧版用 `HashMap` 遍历 + 逐个 `String::replace`，有两个隐患：
//! 1. 替换顺序不确定，若某个变量值里恰好含 `{{other_key}}` 会被二次替换（模板注入）；
//! 2. 模板里写错占位符名（或代码少传一个变量）不会报错，字面量 `{{source_lang}}`
//!    会原样进入提示词——这个 bug 在 v0.1 真实发生过。
//!
//! 现在改为**单遍扫描替换**（值不再被二次解析），并做双向校验：
//! - 模板里出现未知占位符 -> 直接报错；
//! - 代码传了模板没用的变量 -> warn（多半是拼写错误）。

use anyhow::{anyhow, Context, Result};
use log::{debug, warn};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

pub struct PromptStore {
    dir: PathBuf,
    /// 逐句质检会高频渲染同一模板，缓存文件内容避免每句一次磁盘 IO
    cache: RefCell<HashMap<String, String>>,
}

impl PromptStore {
    pub fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_path_buf(),
            cache: RefCell::new(HashMap::new()),
        }
    }

    fn load(&self, file_name: &str) -> Result<String> {
        if let Some(t) = self.cache.borrow().get(file_name) {
            return Ok(t.clone());
        }
        let path = self.dir.join(file_name);
        let template = fs::read_to_string(&path)
            .with_context(|| format!("读取提示词失败: {:?}（提示词目录 {:?}）", path, self.dir))?;
        self.cache
            .borrow_mut()
            .insert(file_name.to_string(), template.clone());
        Ok(template)
    }

    /// 模板中出现的占位符名集合。
    fn placeholders(template: &str) -> HashSet<String> {
        let chars: Vec<char> = template.chars().collect();
        let mut out = HashSet::new();
        let mut i = 0;
        while i + 1 < chars.len() {
            if chars[i] == '{' && chars[i + 1] == '{' {
                if let Some(end) = find_close(&chars, i + 2) {
                    let key: String = chars[i + 2..end].iter().collect();
                    let key = key.trim();
                    if !key.is_empty() && !key.contains('{') {
                        out.insert(key.to_string());
                    }
                    i = end + 2;
                    continue;
                }
            }
            i += 1;
        }
        out
    }

    pub fn render(&self, file_name: &str, vars: &HashMap<String, String>) -> Result<String> {
        let template = self.load(file_name)?;

        let used = Self::placeholders(&template);
        for key in used.iter() {
            if !vars.contains_key(key) {
                return Err(anyhow!(
                    "提示词 {} 需要变量 {{{{{}}}}}，但代码没有提供（已提供: {}）",
                    file_name,
                    key,
                    sorted_keys(vars)
                ));
            }
        }
        for key in vars.keys() {
            if !used.contains(key) {
                warn!(
                    "提示词 {} 未使用变量 {{{{{}}}}}（模板里可能写错了占位符名）",
                    file_name, key
                );
            }
        }

        // 单遍扫描：值原样写入，不会被再次当作模板解析
        let chars: Vec<char> = template.chars().collect();
        let mut out = String::with_capacity(template.len() + 128);
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '{' && i + 1 < chars.len() && chars[i + 1] == '{' {
                if let Some(end) = find_close(&chars, i + 2) {
                    let key: String = chars[i + 2..end].iter().collect();
                    let key = key.trim();
                    if let Some(v) = vars.get(key) {
                        out.push_str(v);
                        i = end + 2;
                        continue;
                    }
                }
            }
            out.push(chars[i]);
            i += 1;
        }
        debug!("渲染 {} 完成：{} 字", file_name, out.chars().count());
        Ok(out)
    }
}

/// 从 `start` 起找配对的 `}}`，返回其起始下标。
fn find_close(chars: &[char], start: usize) -> Option<usize> {
    let mut i = start;
    while i + 1 < chars.len() {
        if chars[i] == '}' && chars[i + 1] == '}' {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn sorted_keys(vars: &HashMap<String, String>) -> String {
    let mut k: Vec<&String> = vars.keys().collect();
    k.sort();
    k.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with(name: &str, body: &str) -> PromptStore {
        let dir = std::env::temp_dir().join(format!("qsa-prompts-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(name), body).unwrap();
        PromptStore::new(&dir)
    }

    fn vars(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn substitutes_all_placeholders() {
        let s = store_with("a.txt", "从 {{src}} 到 {{dst}}：{{text}}");
        let out = s
            .render(
                "a.txt",
                &vars(&[("src", "日语"), ("dst", "中文"), ("text", "こんにちは")]),
            )
            .unwrap();
        assert_eq!(out, "从 日语 到 中文：こんにちは");
    }

    #[test]
    fn unknown_placeholder_is_an_error() {
        let s = store_with("b.txt", "{{source_lang}} -> {{target_lang}}");
        let e = s
            .render("b.txt", &vars(&[("source_lang", "ja")]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("target_lang"), "{e}");
    }

    #[test]
    fn values_are_not_rescanned() {
        // 值里含 {{dst}} 不应被二次替换（防模板注入）
        let s = store_with("c.txt", "A={{a}} B={{b}}");
        let out = s
            .render("c.txt", &vars(&[("a", "{{b}}"), ("b", "X")]))
            .unwrap();
        assert_eq!(out, "A={{b}} B=X");
    }

    #[test]
    fn json_examples_with_single_braces_are_left_alone() {
        let s = store_with("d.txt", "输出 {\"i\":1,\"t\":\"x\"} 与 {{text}}");
        let out = s.render("d.txt", &vars(&[("text", "T")])).unwrap();
        assert_eq!(out, "输出 {\"i\":1,\"t\":\"x\"} 与 T");
    }

    #[test]
    fn placeholder_key_may_contain_spaces() {
        let s = store_with("e.txt", "[{{ text }}]");
        let out = s.render("e.txt", &vars(&[("text", "T")])).unwrap();
        assert_eq!(out, "[T]");
    }

    #[test]
    fn missing_file_reports_dir() {
        let s = PromptStore::new(Path::new("./definitely-not-here"));
        let e = s.render("nope.txt", &vars(&[])).unwrap_err().to_string();
        assert!(e.contains("读取提示词失败"), "{e}");
    }

    #[test]
    fn placeholders_extraction() {
        let p = PromptStore::placeholders("{{a}} x {{ b }} y {not} z {{a}}");
        assert!(p.contains("a"));
        assert!(p.contains("b"));
        assert_eq!(p.len(), 2);
    }
}
