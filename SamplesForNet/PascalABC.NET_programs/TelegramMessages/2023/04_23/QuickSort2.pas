// Быстрая сортировка Ч. Хоара
/// Разделение a[i]..a[j] на части a[i]..a[q] <= a[q+1]..a[j] 
function Partition(a: array of integer; i,j: integer): integer;
begin
  var x := a[i];
  while True do begin
    while a[i]<x do
      i += 1;
    while a[j]>x do
      j -= 1;
    if i>=j then 
    begin
      Result := j;
      exit;
    end;
    Swap(a[i],a[j]);
    i += 1;
    j -= 1;
  end;
end;
  
/// Сортировка частей
procedure QuickSort(a: array of integer; left,right: integer);
begin
  if left >= right then exit;
  var q := Partition(a,left,right);
  QuickSort(a,left,q);
  QuickSort(a,q+1,right);
end;

const n = 20;

begin
  var a := ArrRandom(n);
  Println('До сортировки: ');
  Writeln(a);
  QuickSort(a,0,a.Length-1);
  Println('После сортировки: ');
  Println(a);
end.
