param(
    [string]$PythonExe = $null
)

# Determine repository root (one level up from this scripts folder)
$repoRoot = Split-Path -Parent $PSScriptRoot

if (-not $PythonExe) {
    $defaultPython = Join-Path $repoRoot 'venv\Scripts\python.exe'
    if (Test-Path $defaultPython) { $PythonExe = $defaultPython } else { $PythonExe = 'python' }
}

# Path to the client script (relative to the repository root)
$clientScript = Join-Path $repoRoot 'federated\client.py'

$names = @('Client-1','Client-2','Client-3','Client-4')

foreach ($name in $names) {
    $cmd = "Set-Location -Path '$repoRoot'; & '$PythonExe' '$clientScript' --name $name"
    Start-Process powershell -ArgumentList "-NoExit","-Command","$cmd"
    Start-Sleep -Milliseconds 300
}

Write-Output "Launched $($names.Count) client windows. Close them when finished."