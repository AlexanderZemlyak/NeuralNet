begin
  var s := 'abracadabra';
  var s1 := 'bra';
  var n := s1.Length;
  for var i:=1 to s.Length do
    if s?[i:i+n] = s1 then
      print(i);
end.