$ErrorActionPreference = 'Stop'
$taskSource = Join-Path $PSScriptRoot 'final'
$taskDestination = 'D:\Desktop\待交付\AI模块使用维护及培训资料'
$taskFiles = Get-ChildItem -LiteralPath $taskSource -File | Sort-Object Name
if ($taskFiles.Count -ne 8) { throw 'Expected exactly eight delivery files.' }
if (Test-Path -LiteralPath $taskDestination) { throw 'Destination already exists; no files were changed.' }
New-Item -ItemType Directory -Path $taskDestination | Out-Null
foreach ($taskFile in $taskFiles) {
    $taskTarget = Join-Path $taskDestination $taskFile.Name
    Copy-Item -LiteralPath $taskFile.FullName -Destination $taskTarget
    $taskExpectedHash = (Get-FileHash -LiteralPath $taskFile.FullName -Algorithm SHA256).Hash
    $taskActualHash = (Get-FileHash -LiteralPath $taskTarget -Algorithm SHA256).Hash
    if ($taskExpectedHash -ne $taskActualHash) { throw ('Copy verification failed: ' + $taskFile.Name) }
}
Get-ChildItem -LiteralPath $taskDestination -File | Sort-Object Name | Select-Object Name,Length
