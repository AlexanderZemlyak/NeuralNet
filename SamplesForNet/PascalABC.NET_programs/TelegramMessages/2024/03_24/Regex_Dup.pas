##
var s := 'он  он сказал, что   что это правильный ответ';
foreach var m in s.Matches('\b(\w+)\s+(\1)\b') do
   Println($'Дубликат "{m.Groups[1]}" в позициях {m.Groups[1].Index} и {m.Groups[2].Index}.')