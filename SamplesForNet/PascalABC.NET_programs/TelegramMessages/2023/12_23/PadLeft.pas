uses School;

function ip2int(s: string): integer;
begin
  var sBin := s.Split('.').Select(x -> BinFormat(x.ToInteger,0)).JoinToString('');
  Println(sBin);
  Result := Dec(sBin,2);
end;

begin
  var s := '127.0.0.1';
  print(ip2int( s ));
  //Print(BinFormat(12,0));
  //var a := 12;
  //a.ToString.PadLeft(8,'0').Print;
  //Print($'{a:D8}');
end.