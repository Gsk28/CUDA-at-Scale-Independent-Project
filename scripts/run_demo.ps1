$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$BuildDir = Join-Path $RootDir "build"
$InputDir = Join-Path $RootDir "data\input"
$OutputDir = Join-Path $RootDir "data\output"
$ProofDir = Join-Path $RootDir "proof\latest"

New-Item -ItemType Directory -Force -Path $ProofDir | Out-Null
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $InputDir, $OutputDir, (Join-Path $ProofDir "outputs")
New-Item -ItemType Directory -Force -Path $InputDir, $OutputDir, (Join-Path $ProofDir "outputs") | Out-Null

python (Join-Path $RootDir "scripts\generate_dataset.py") `
  --output $InputDir `
  --count 256 `
  --width 256 `
  --height 256 `
  --seed 7 | Tee-Object -FilePath (Join-Path $ProofDir "dataset.log")

cmake -S $RootDir -B $BuildDir -DCMAKE_BUILD_TYPE=Release
cmake --build $BuildDir --config Release

$Executable = Join-Path $BuildDir "Release\cuda_batch_edges.exe"
if (-not (Test-Path $Executable)) {
  $Executable = Join-Path $BuildDir "cuda_batch_edges.exe"
}

& $Executable `
  --input $InputDir `
  --output $OutputDir `
  --threshold 88 `
  --warmup 1 | Tee-Object -FilePath (Join-Path $ProofDir "run.log")

Copy-Item (Join-Path $OutputDir "summary.csv") (Join-Path $ProofDir "summary.csv") -Force
Get-ChildItem $OutputDir -Filter "synthetic_*.pgm" |
  Sort-Object Name |
  Select-Object -First 12 |
  Copy-Item -Destination (Join-Path $ProofDir "outputs") -Force

python (Join-Path $RootDir "scripts\make_contact_sheet.py") `
  --input (Join-Path $ProofDir "outputs") `
  --output (Join-Path $ProofDir "contact_sheet.png") `
  --limit 12

Write-Host "Proof artifacts written to $ProofDir"
