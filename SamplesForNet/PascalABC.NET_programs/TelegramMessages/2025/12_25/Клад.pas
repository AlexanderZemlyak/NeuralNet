begin
  var s := 'Клад зарыт на [юго-западе] под [пальмой]';
  var searchWord := 'скрыто';
  
  var searchPos := 0;  
  
  while True do
  begin
    var startPos := s.IndexOf('[', searchPos);  
    if startPos = -1 then break;                
    
    var endPos := s.IndexOf(']', startPos + 1); 
    if endPos = -1 then break;                  
    
    var removeStart := startPos + 1;
    var length := endPos - startPos - 1;
    
    s := s.Remove(removeStart, length);
    
    s := s.Insert(removeStart, searchWord);
    
    searchPos := removeStart + searchWord.Length;
  end;
  
  Println(s);
end.