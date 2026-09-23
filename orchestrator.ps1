$env:PYTHONIOENCODING="utf-8"
$env:PYTHONPATH="C:\CODE2026\CASSIproy2026\CASSIproy2026\DSLAnet_1.0"
$pythonPath = "C:\Users\gustavo.veracoechea\AppData\Local\Programs\Python\Python310\python.exe"

# Ablation Strategy Parameters
$variants = @("baseline", "spatcab", "speccab", "fullcab")
$blocksList = @(4, 8, 12, 16, 20)
$epochs = 400

Write-Host "====================================================="
Write-Host "   Starting DSLAnet 1.0 Ablation Sweep Orchestrator  "
Write-Host "====================================================="
Write-Host "Total combinations to train: $(($variants.Length) * ($blocksList.Length))"
Write-Host ""

cd "C:\CODE2026\CASSIproy2026\CASSIproy2026\DSLAnet_1.0"

foreach ($v in $variants) {
    foreach ($b in $blocksList) {
        Write-Host "[*] Launching Training: Variant='$v' | Blocks=$b | Epochs=$epochs"
        
        # Execute the python training script
        $command = "& `"$pythonPath`" train_sweep.py --variant $v --blocks $b --epochs $epochs"
        Invoke-Expression $command
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[!] Error encountered during $v with $b blocks. Exiting orchestrator to prevent cascading failures." -ForegroundColor Red
            exit $LASTEXITCODE
        }
        
        Write-Host "[+] Completed: $v with $b blocks." -ForegroundColor Green
        Write-Host "[-] Cooling down GPU memory for 5 seconds..."
        Start-Sleep -Seconds 5
    }
}

Write-Host "====================================================="
Write-Host "   Ablation Study Completed Successfully!            "
Write-Host "====================================================="
