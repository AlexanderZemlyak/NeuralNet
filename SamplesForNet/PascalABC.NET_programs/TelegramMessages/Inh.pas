{ Дан массив. Вывести все элементы, бОльшие своего левого соседа }  
begin 
  var a := ArrRandomInteger(10);
  a.Println;
  var n := a.Length;
// Способ 1
  for var i:=1 to n-1 do
    if a[i] > a[i-1] then
      Print(a[i]);
  Println;
// Способ 2
  for var i:=0 to n-2 do
    if a[i+1] > a[i] then
      Print(a[i+1]);
end.
