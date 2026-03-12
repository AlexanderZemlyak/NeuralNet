begin
  var s := 'Клад зарыт на [юго-западе] под [пальмой]';
  var replaceStr := 'скрыто';
  
  var startPos := s.IndexOf('[');
  while startPos <> -1 do
  begin
    var endPos := s.IndexOf(']', startPos + 1);
    if endPos = -1 then break;
    
    s := s.Remove(startPos + 1, endPos - startPos - 1);
    s := s.Insert(startPos + 1, replaceStr);
    
    startPos := s.IndexOf('[', endPos);
  end;
  
  Println(s);
end.
