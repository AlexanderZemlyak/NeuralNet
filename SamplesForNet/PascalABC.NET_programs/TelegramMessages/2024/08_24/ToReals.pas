begin
  var nfi := NumberFormat(DecimalSeparator := ',');
  // В файле ar.txt пропускаются знаки табуляции и символы перехода на новую строку
  ReadAllText('ar.txt').ToReals(nfi).Println;
  ReadAllText('ar.txt').ToReals(NumberFormat(',')).Println;
  ReadAllText('ar.txt').ToReals(',').Println;
  var s := '2,4';
  s.ToReal(',').Print;
  s.ToReal(nfi).Print;
  StrToReal(s,nfi).Print;
  StrToReal(s,',').Print;
  
  SetDecimalSeparator(',');
  StrToReal(s).Print;
end.