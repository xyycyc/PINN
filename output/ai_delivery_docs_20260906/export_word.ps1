$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskFinal = Join-Path $taskRoot 'final'
$taskWord = New-Object -ComObject Word.Application
$taskWord.Visible = $false
$taskWord.DisplayAlerts = 0
try {
    $taskFiles = Get-ChildItem -LiteralPath $taskFinal -Filter '*.docx' | Sort-Object Name
    foreach ($taskFile in $taskFiles) {
        $taskDoc = $null
        try {
            $taskDoc = $taskWord.Documents.Open($taskFile.FullName, $false, $true, $false)
            $taskDoc.Repaginate()
            $taskPages = $taskDoc.ComputeStatistics(2)
            if ($taskFile.Name.StartsWith('01_')) {
                $taskPdfDir = Join-Path $taskRoot 'qa\original_word'
                New-Item -ItemType Directory -Path $taskPdfDir -Force | Out-Null
                $taskPdf = Join-Path $taskPdfDir ($taskFile.BaseName + '.pdf')
            } else {
                $taskPdf = Join-Path $taskFinal ($taskFile.BaseName + '.pdf')
            }
            $taskDoc.ExportAsFixedFormat($taskPdf, 17)
            Write-Output ($taskFile.Name + ' PAGES=' + $taskPages)
        } finally {
            if ($null -ne $taskDoc) {
                $taskDoc.Close(0)
                [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($taskDoc)
            }
        }
    }
} finally {
    $taskWord.Quit()
    [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($taskWord)
}
