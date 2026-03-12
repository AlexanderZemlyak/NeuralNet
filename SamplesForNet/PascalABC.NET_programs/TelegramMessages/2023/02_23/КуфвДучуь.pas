begin
  loop 10 do
  begin
    var lex := ReadLexem;
    if lex.IsReal then
      Println('вещ',lex.ToReal)
    else if lex.IsReal then
      Println('цел',lex.ToInteger)
    else Println('лексема')
  end;
end.