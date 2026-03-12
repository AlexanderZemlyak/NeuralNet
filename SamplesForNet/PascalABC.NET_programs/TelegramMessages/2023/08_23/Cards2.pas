begin
  var bb := Encoding.Unicode.GetBytes('🂡');
  //var bb := Arr&<byte>(60,216,161,220);
  var Cards := LstStr;
  for var i := 0 to 4*16-1 do
  begin
    var s := Encoding.Unicode.GetString(bb);
    if i mod 16 not in |11,14,15| then
      Cards += s;
    bb[2] += 1;
  end;
  Cards.Println;
  Println;
  Shuffle(Cards);
  Cards.Println;
end.