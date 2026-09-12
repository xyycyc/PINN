$ErrorActionPreference = 'Stop'
$taskFinal = Join-Path $PSScriptRoot 'final'
$taskWord = New-Object -ComObject Word.Application
$taskWord.Visible = $false
$taskWord.DisplayAlerts = 0
try {
    foreach ($taskFile in (Get-ChildItem -LiteralPath $taskFinal -Filter '*.docx')) {
        $taskDoc = $null
        try {
            $taskDoc = $taskWord.Documents.Open($taskFile.FullName, $false, $true, $false)
            $taskDoc.Repaginate()
            $taskPages = $taskDoc.ComputeStatistics(2)
            $taskPdf = Join-Path $taskFinal ($taskFile.BaseName + '.pdf')
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

