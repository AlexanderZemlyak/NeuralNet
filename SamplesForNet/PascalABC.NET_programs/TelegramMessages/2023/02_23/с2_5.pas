begin
  var a := ArrRandom(10,2,5);
  a.Println;
  var cc := |0|*6;
  foreach var x in a do
    if x in 2..5 then
      cc[x] += 1;
  for var i:=2 to 5 do
    Println('Оценка',i,'-',cc[i],'шт');
end.