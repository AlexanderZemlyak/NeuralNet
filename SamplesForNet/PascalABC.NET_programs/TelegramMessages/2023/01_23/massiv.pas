begin
  var kol := 0;
  var massiv := Arr(1,2,3,5,2);
  var dlina := massiv.Length;
  for var i:=0 to dlina - 1 do
    massiv[i] := massiv[i] + 1;
  
  var summa := 0;
  for var i:=0 to dlina - 1 do
    summa := summa + massiv[i];
  var srednee := summa/dlina;
  Print(srednee)
end.