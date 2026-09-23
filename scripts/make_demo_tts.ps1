# Regenerates demo/tts/*.wav and demo/inputs.txt with the Windows speech synthesiser.
#
#   powershell -ExecutionPolicy Bypass -File scripts/make_demo_tts.ps1
#
# Sentences come from demo/tts_lines.tsv (id<TAB>voice<TAB>text, UTF-8). The default
# lines use the Huihui (zh-CN) and Zira (en-US) voices that ship with Windows and mix
# assistant commands with TV-style sentences on purpose: both classes are rendered
# from the same pool, so content never decides the label.
param(
  [string]$Lines = (Join-Path $PSScriptRoot '..\demo\tts_lines.tsv'),
  [string]$OutDir = (Join-Path $PSScriptRoot '..\demo\tts')
)

Add-Type -AssemblyName System.Speech
$null = New-Item -ItemType Directory -Force -Path $OutDir
$OutDir = (Resolve-Path $OutDir).Path
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
  24000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)

$out = @('# Real TTS from the Windows speech synthesiser, 24 kHz mono.',
         '# One utterance per line: path<TAB>text, relative to this file.')
$count = 0
foreach ($row in [System.IO.File]::ReadAllLines((Resolve-Path $Lines).Path, [System.Text.Encoding]::UTF8)) {
  if (-not $row.Trim() -or $row.StartsWith('#')) { continue }
  $id, $voice, $text = $row.Split("`t", 3)
  $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
  $synth.SelectVoice($voice)
  $synth.SetOutputToWaveFile((Join-Path $OutDir ($id + '.wav')), $format)
  $synth.Speak($text)
  $synth.Dispose()
  $out += ('tts/' + $id + '.wav' + "`t" + $text)
  $count += 1
}
$listPath = Join-Path (Split-Path $OutDir -Parent) 'inputs.txt'
[System.IO.File]::WriteAllLines($listPath, $out, (New-Object System.Text.UTF8Encoding($false)))
Write-Output "wrote $count utterances to $OutDir and $listPath"
