begin
  // Описание массива
  var a: array of integer;
  // Выделение памяти
  a := new integer[5];
  // Заполнение 
  a[0] := 5; 
  a[1] := 3; 
  a[2] := 2; 
  a[3] := 5; 
  a[4] := 4;
  // Вывод
  for var i:=0 to a.Length-1 do 
    Print(a[i])
end.