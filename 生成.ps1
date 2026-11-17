param(
    [Parameter(Mandatory=$true)][string]$Text,
    [string]$Context = '',
    [string]$Output = (Join-Path $PSScriptRoot 'results\synthesis.wav'),
    [switch]$Compare
)
$py = 'F:\GPT-SoVITS\venv\Scripts\python.exe'
$arguments = @('-X','utf8',(Join-Path $PSScriptRoot 'synthesize.py'),'--text',$Text,'--context',$Context,'--output',$Output)
if ($Compare) { $arguments += '--compare' }
Push-Location -LiteralPath $PSScriptRoot
try {
    & $py @arguments
    if ($LASTEXITCODE) { throw "Synthesis failed (exit $LASTEXITCODE)" }
} finally { Pop-Location }
