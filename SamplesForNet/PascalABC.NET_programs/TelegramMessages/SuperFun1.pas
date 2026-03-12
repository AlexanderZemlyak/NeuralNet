function operator*<T,T1,T2>(f: T1 -> T2; g: T -> T1): T -> T2; extensionmethod := x -> f(g(x));

begin
  var fun := Sin * Cos;
  Print(fun(1));
end.