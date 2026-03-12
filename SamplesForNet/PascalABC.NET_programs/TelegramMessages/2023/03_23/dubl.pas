##
var s := 'он  он сказал, что    что это правильный ответ';
s.Matches('(\w+)\s+(\1)')
 .PrintLines(m -> $'Дубликат {m.Groups[1].Value} в позициях {m.Groups[1].Index} и {m.Groups[2].Index}');
