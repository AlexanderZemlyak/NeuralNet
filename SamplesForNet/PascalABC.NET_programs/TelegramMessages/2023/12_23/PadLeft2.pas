uses School;

begin
  var a := 12;
  a.ToString.PadLeft(8,'0').Println;
  Println($'{a:D8}');
  Println(BinFormat(a,0));
end.