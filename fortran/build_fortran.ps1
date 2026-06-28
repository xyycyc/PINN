param(
    [string]$Compiler = "gfortran"
)

$ErrorActionPreference = "Stop"
$SourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceFile = Join-Path $SourceDir "ai_numeric.f90"
$OutputFile = Join-Path $SourceDir "ai_numeric.dll"

if (-not (Get-Command $Compiler -ErrorAction SilentlyContinue)) {
    throw "Fortran compiler '$Compiler' was not found in PATH."
}

& $Compiler `
    -O2 `
    -shared `
    -static-libgfortran `
    -static-libgcc `
    "-J$SourceDir" `
    -o $OutputFile `
    $SourceFile

if ($LASTEXITCODE -ne 0) {
    throw "Fortran compilation failed with exit code $LASTEXITCODE."
}

Write-Host "Built Fortran library: $OutputFile"
