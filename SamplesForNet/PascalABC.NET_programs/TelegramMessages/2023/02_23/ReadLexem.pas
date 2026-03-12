begin
  loop 10 do
  begin
    var lex := ReadLexem;
    if lex.IsInteger then
      Println('цел',lex.ToInteger)
    else if lex.IsReal then
      Println('вещ',lex.ToReal)
    else Println('лексема')
  end;
end.