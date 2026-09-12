$ErrorActionPreference = 'Stop'
$taskDestination = 'D:\Desktop\待交付\AI模块使用维护及培训资料'
$taskSource = Join-Path $PSScriptRoot 'renamed'
$taskNames = [ordered]@{
    '01_AI模块程序使用说明' = '程序使用说明'
    '02_AI模块维护手册' = '维护手册'
    '03_AI模块程序调试维修方法' = '程序调试维修方法'
    '04_AI模块培训记录' = '培训记录'
}
foreach ($taskEntry in $taskNames.GetEnumerator()) {
    foreach ($taskExt in @('.docx','.pdf')) {
        $taskNewName = $taskEntry.Value + $taskExt
        $taskFrom = Join-Path $taskSource $taskNewName
        $taskTo = Join-Path $taskDestination $taskNewName
        if (Test-Path -LiteralPath $taskTo) { throw ('New name already exists: ' + $taskNewName) }
        if (-not (Test-Path -LiteralPath $taskFrom)) { throw ('Source missing: ' + $taskNewName) }
    }
}
foreach ($taskEntry in $taskNames.GetEnumerator()) {
    foreach ($taskExt in @('.docx','.pdf')) {
        $taskFrom = Join-Path $taskSource ($taskEntry.Value + $taskExt)
        $taskTo = Join-Path $taskDestination ($taskEntry.Value + $taskExt)
        Copy-Item -LiteralPath $taskFrom -Destination $taskTo
        if ((Get-FileHash -LiteralPath $taskFrom).Hash -ne (Get-FileHash -LiteralPath $taskTo).Hash) { throw 'Copy verification failed.' }
    }
}
foreach ($taskEntry in $taskNames.GetEnumerator()) {
    foreach ($taskExt in @('.docx','.pdf')) {
        $taskOld = Join-Path $taskDestination ($taskEntry.Key + $taskExt)
        Remove-Item -LiteralPath $taskOld
    }
}
Get-ChildItem -LiteralPath $taskDestination -File | Sort-Object Name | Select-Object Name,Length
