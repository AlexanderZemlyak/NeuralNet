uses School;

begin
  var pp := Primes(500);
  for var i:=0 to pp.Count-1 do
    if '5' in pp[i].ToString then
      Println(pp[i],i+1)
end.