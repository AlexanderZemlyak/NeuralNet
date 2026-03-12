function Perm<T>(a: array of T; n: integer): sequence of array of T;
begin
  if n = 1 then
    yield Copy(a)
  else
    for var i := 0 to n-1 do
    begin
      Swap(a[i],a[n-1]);
      yield sequence Perm(a,n-1);
      Swap(a[i],a[n-1]);
    end;
end;

begin
  var a := Arr(1..4);
  Perm(a,4).Print
end.