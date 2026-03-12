function operator*(f,g: real -> real): real -> real; extensionmethod 
  := x -> f(g(x));

begin
  var fun := Sin * Cos;
  Println(fun(1));
  Println((Cos * Sin)(1));
end.