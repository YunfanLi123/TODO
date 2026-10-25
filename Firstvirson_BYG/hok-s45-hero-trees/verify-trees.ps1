$ErrorActionPreference = 'Stop'
$p = '<本机目录>'
$enc = New-Object System.Text.UTF8Encoding($false)
$b = [System.IO.File]::ReadAllBytes($p)
$t = [System.IO.File]::ReadAllText($p, $enc)
$lines = $t -split "`r?`n"

$patPre  = '^\s*(└─|├─) 前提：'
$patAct  = '^\s*(└─|├─) 动作：|^\s*│\s+动作：|^\s{3,6}动作：'
$patCon  = '^\s*(└─|├─) 后果：|^\s*│\s+后果：|^\s{3,6}后果：'
$patNode = '^(0:00|2:00|4:00|6:00|10:00) 节点$|^12:00 之后$'

Write-Output ("BYTES=" + $b.Length)
Write-Output ("BOM=" + ($b[0] -eq 0xEF -and $b[1] -eq 0xBB -and $b[2] -eq 0xBF))
Write-Output ("LINES=" + $lines.Count)

$idxPre = @(); $idxAct = @(); $idxCon = @(); $idxNode = @()
for ($i = 0; $i -lt $lines.Count; $i++) {
  if ($lines[$i] -match $patPre)  { $idxPre  += $i }
  if ($lines[$i] -match $patAct)  { $idxAct  += $i }
  if ($lines[$i] -match $patCon)  { $idxCon  += $i }
  if ($lines[$i] -match $patNode) { $idxNode += $i }
}
Write-Output ("NODE_HEADERS=" + $idxNode.Count)
Write-Output ("PRE=" + $idxPre.Count + " ACT=" + $idxAct.Count + " CON=" + $idxCon.Count)

$badTag = @($idxCon | Where-Object { $lines[$_] -notmatch '【[^】]+】\s*$' })
Write-Output ("CON_WITHOUT_TAG=" + $badTag.Count)

$badBranch = @()
foreach ($i in $idxPre) {
  $j = $i + 1; $hasAct = 0; $conCount = 0
  while ($j -lt $lines.Count) {
    if ($lines[$j] -match $patPre) { break }
    if ($lines[$j] -match $patNode) { break }
    if ($lines[$j] -match $patAct) { $hasAct++ }
    if ($lines[$j] -match $patCon) { $conCount++ }
    $j++
  }
  if ($hasAct -lt 1 -or $conCount -ne 1) { $badBranch += ("L" + ($i+1) + " act=" + $hasAct + " con=" + $conCount + " :: " + $lines[$i]) }
}
Write-Output ("MALFORMED_BRANCHES=" + $badBranch.Count)
$badBranch | Select-Object -First 5 | ForEach-Object { Write-Output ("  BAD_BRANCH: " + $_) }

$shape = @($lines | Where-Object {
  $_ -match '(└─|├─) 前提：' -and $_ -notmatch '^\s*(└─|├─) 前提：\S'
})
Write-Output ("BAD_PREMISE_SHAPE=" + $shape.Count)
$shape | Select-Object -First 5 | ForEach-Object { Write-Output ("  SHAPE: " + $_) }

$md = @($lines | Where-Object { $_ -match '[*#|`]' })
Write-Output ("MARKDOWN_LINES=" + $md.Count)
$eq = @($lines | Where-Object { $_ -match '^={5,}$' })
Write-Output ("EQ_SEPARATOR_LINES=" + $eq.Count)

$heroes = @('夏侯惇','吕布','关羽','马超','铠','老夫子','狂铁','夏洛特','蒙恬','曹操','司空震','芈月','亚连','姬小满','达摩','花木兰','哪吒','盘古','杨戬','橘右京','白起','廉颇','项羽','猪八戒','程咬金','刘禅','钟无艳','元歌','兰陵王','大司命','墨子','刘邦','牛魔')
$expect = @('0:00 节点','2:00 节点','4:00 节点','6:00 节点','10:00 节点','12:00 之后')
Write-Output ("HEROES_EXPECTED=" + $heroes.Count)
$missing = @(); $badNode = @(); $noLoc = @(); $orderBad = @(); $lt2 = @(); $tierMiss = @(); $mismatch = @(); $noRef = @()
foreach ($h in $heroes) {
  $idx = -1
  for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -ceq $h) { $idx = $i; break } }
  if ($idx -lt 0) { $missing += $h; continue }
  $end = $lines.Count
  for ($j = $idx + 1; $j -lt $lines.Count; $j++) { if ($heroes -contains $lines[$j]) { $end = $j; break } }
  $loc = $lines[$idx+1]
  if ($loc -notmatch '^  ') { $noLoc += $h }
  if ($loc -notmatch '梯队：T[0-3]') { $tierMiss += $h }
  $seq = @(); $cnt = @{}; $cur = ''
  for ($j = $idx; $j -lt $end; $j++) {
    if ($lines[$j] -match $patNode) { $seq += $lines[$j]; $cur = $lines[$j]; if (-not $cnt.ContainsKey($cur)) { $cnt[$cur] = 0 } }
    elseif ($lines[$j] -match $patPre) { $cnt[$cur] = $cnt[$cur] + 1 }
  }
  if ($seq.Count -ne 6) { $badNode += ($h + '=' + $seq.Count) }
  if (($seq -join '|') -ne ($expect -join '|')) { $orderBad += $h }
  foreach ($k in $cnt.Keys) { if ($cnt[$k] -lt 2) { $lt2 += ($h + '|' + $k + '=' + $cnt[$k]) } }
  $hPre = 0; $hCon = 0
  for ($j = $idx; $j -lt $end; $j++) { if ($lines[$j] -match $patPre) { $hPre++ }; if ($lines[$j] -match $patCon) { $hCon++ } }
  if ($hPre -ne $hCon) { $mismatch += ($h + ' pre=' + $hPre + ' con=' + $hCon) }
  $nodeIdx = @()
  for ($j = $idx; $j -lt $end; $j++) { if ($lines[$j] -match $patNode) { $nodeIdx += $j } }
  for ($n = 0; $n -lt $nodeIdx.Count - 1; $n++) {
    $s = $nodeIdx[$n]; $e = $nodeIdx[$n+1]
    $tags = @()
    for ($j = $s; $j -lt $e; $j++) { if ($lines[$j] -match $patCon) { foreach ($m in [regex]::Matches($lines[$j], '【([^】]+)】')) { $tags += $m.Groups[1].Value } } }
    $hi = [Math]::Min($nodeIdx[$n+1] + 90, $lines.Count - 1)
    $nextText = [string]::Join(' ', $lines[$nodeIdx[$n+1]..$hi])
    $hit = $false
    foreach ($tg in $tags) { if ($nextText.Contains($tg)) { $hit = $true } }
    if (-not $hit) { $noRef += ($h + '@' + $lines[$s]) }
  }
}
Write-Output ("MISSING_HEROES=" + ($missing -join ','))
Write-Output ("HEROES_BAD_NODE_COUNT=" + ($badNode -join ','))
Write-Output ("HEROES_BAD_NODE_ORDER=" + ($orderBad -join ','))
Write-Output ("HEROES_MISSING_LOC_LINE=" + ($noLoc -join ','))
Write-Output ("HEROES_MISSING_TIER=" + ($tierMiss -join ','))
Write-Output ("NODES_WITH_LT2_BRANCHES=" + ($lt2 -join ' ; '))
Write-Output ("HEROES_PRE_CON_MISMATCH=" + ($mismatch -join ' ; '))
Write-Output ("NODES_WITHOUT_TAG_REUSE=" + ($noRef -join ' ; '))

$banned = @('最强','必选','版本答案','最重要','划算','很赚','碾压','无脑','闭眼选','完胜','版本最强')
$hits = @()
for ($i = 0; $i -lt $lines.Count; $i++) { foreach ($w in $banned) { if ($lines[$i].Contains($w)) { $hits += ("L" + ($i+1) + " [" + $w + "]") } } }
Write-Output ("BANNED_WORD_HITS=" + $hits.Count)

$econ = @('经济差','等级差','评分','局势','劣势','优势局面')
$ehits = @()
for ($i = 0; $i -lt $lines.Count; $i++) { foreach ($w in $econ) { if ($lines[$i].Contains($w)) { $ehits += ("L" + ($i+1) + " [" + $w + "] " + $lines[$i]) } } }
Write-Output ("ECON_PREMISE_HITS=" + $ehits.Count)
$ehits | Select-Object -First 8 | ForEach-Object { Write-Output ("  ECON: " + $_) }
