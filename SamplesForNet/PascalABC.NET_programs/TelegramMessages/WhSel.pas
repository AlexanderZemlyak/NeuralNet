function Filter<T>(Self: sequence of T; cond: T -> boolean): sequence of T; extensionmethod;
begin
  foreach var x in Self do
    if cond(x) then
      yield x
end;

function Convert<T,T1>(Self: sequence of T; conv: T -> T1): sequence of T1; extensionmethod;
begin
  foreach var x in Self do
    yield conv(x)
end;

begin
  var a := Arr(1..9);
  a.Filter(x -> x.IsOdd).Convert(x -> x * x).Print;
end.