# 一键下载运行所需的全部模型到程序目录的 models\ 下：
#   1. Qwen3-ASR 0.6B int8（sherpa-onnx 格式，GitHub Release）
#   2. Silero VAD（GitHub Release）
#   3. Qwen3-1.7B-Q8_0.gguf（HuggingFace，可用 -HfMirror 切换镜像）
#
# 用法（在 PowerShell 中）：
#   powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1
#   国内网络推荐：
#   powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1 -HfMirror https://hf-mirror.com
#   只要 ASR/VAD，不要 LLM：
#   powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1 -SkipLlm

param(
    [string]$HfMirror = "https://huggingface.co",
    [switch]$SkipLlm
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir
$models = Join-Path $root "models"
New-Item -ItemType Directory -Force -Path $models | Out-Null
$tmp = Join-Path $env:TEMP "qsa-models-dl"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

function Get-File {
    param([string]$Url, [string]$Dest)
    if ((Test-Path $Dest) -and ((Get-Item $Dest).Length -gt 0)) {
        Write-Host "已存在，跳过: $Dest" -ForegroundColor DarkGray
        return
    }
    Write-Host "下载: $Url" -ForegroundColor Cyan
    # curl.exe 为 Windows 10 1803+ 自带；-L 跟随重定向
    & curl.exe -L --fail --retry 3 --retry-delay 2 --progress-bar -o $Dest $Url
    if ($LASTEXITCODE -ne 0) {
        Remove-Item $Dest -ErrorAction SilentlyContinue
        throw "下载失败（curl 退出码 $LASTEXITCODE）: $Url"
    }
}

# ---------- 1. Qwen3-ASR 0.6B int8 ----------
$asrDir = Join-Path $models "sherpa-onnx-qwen3-asr-0.6B-int8"
if (Test-Path (Join-Path $asrDir "encoder.int8.onnx")) {
    Write-Host "ASR 模型已存在，跳过" -ForegroundColor DarkGray
} else {
    $asrTar = Join-Path $tmp "sherpa-onnx-qwen3-asr-0.6B-int8.tar.bz2"
    Get-File "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2" $asrTar
    Write-Host "解压 ASR 模型（压缩包约 880MB，解压后约 1.1GB）..."
    tar -xf $asrTar -C $tmp
    $extracted = Join-Path $tmp "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
    if (-not (Test-Path $extracted)) { throw "解压后未找到预期目录: $extracted" }
    if (Test-Path $asrDir) { Remove-Item -Recurse -Force $asrDir }
    Move-Item $extracted $asrDir
    Write-Host "✔ ASR 模型就绪: $asrDir" -ForegroundColor Green
}

# ---------- 2. Silero VAD ----------
Get-File "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" (Join-Path $models "silero_vad.onnx")
Write-Host "✔ VAD 模型就绪" -ForegroundColor Green

# ---------- 3. Qwen3-1.7B GGUF（翻译 + 逐句质检共用） ----------
if ($SkipLlm) {
    Write-Host "已按 -SkipLlm 跳过 LLM 下载。请自行将任意 Qwen3 GGUF 放入 models\ 子目录。" -ForegroundColor Yellow
} else {
    $llmDir = Join-Path $models "Qwen3-1.7B-GGUF"
    New-Item -ItemType Directory -Force -Path $llmDir | Out-Null
    Get-File "$HfMirror/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf" (Join-Path $llmDir "Qwen3-1.7B-Q8_0.gguf")
    Write-Host "✔ LLM 就绪: $llmDir" -ForegroundColor Green
}

Write-Host ""
Write-Host "全部模型准备完成！目录: $models" -ForegroundColor Green
