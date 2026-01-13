use std::fs;
use std::path::{Path, PathBuf};

pub fn generated_dir_for(base_dir: impl AsRef<Path>) -> PathBuf {
    base_dir
        .as_ref()
        .join("src")
        .join("solver")
        .join("gpu")
        .join("shaders")
        .join("generated")
}

pub fn write_file_if_changed(output_path: impl AsRef<Path>, content: &str) -> std::io::Result<PathBuf> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Ok(existing) = fs::read_to_string(output_path) {
        if existing == content {
            return Ok(output_path.to_path_buf());
        }
    }
    fs::write(output_path, content)?;
    Ok(output_path.to_path_buf())
}

pub fn write_generated_wgsl(
    base_dir: impl AsRef<Path>,
    filename: impl AsRef<Path>,
    wgsl: &str,
) -> std::io::Result<PathBuf> {
    let output_path = generated_dir_for(base_dir).join(filename);
    write_file_if_changed(output_path, wgsl)
}

