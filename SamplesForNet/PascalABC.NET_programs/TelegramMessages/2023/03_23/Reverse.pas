##
  var a := ArrRandom(10);
  a.Println;
  var n := a.Length;
// ---- Способ 1: 3/2*n операций
  for var i:=0 to n div 2 - 1 do
    Swap(a[i],a[n-i-1]);
  a.Println;
// ---- Способ 2: аналогичен 1, без вычисления индексов
  var (x,y) := (0,n - 1);
  while x < y do
  begin
    Swap(a[x],a[y]);
    x += 1;
    y -= 1;
  end;
  a.Println;
// ---- Способ 3: дополнительный массив, n операций
  var b := new integer[n];
  for var i:=0 to n - 1 do
    b[i] := a[n-i-1];
  a := b;
  a.Println;
// ---- Способ 4: стандартная процедура (внутри - способ 2)
  Reverse(a);
  a.Println;
// ---- Способ 5: срезы (внутри - дополнительный массив)
  a := a[::-1];
  a.Println;
