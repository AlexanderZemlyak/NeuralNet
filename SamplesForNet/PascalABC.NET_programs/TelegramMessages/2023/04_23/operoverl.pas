function operator*<T,T1>(a: sequence of T; b: sequence of T1): sequence of (T,T1); extensionmethod
  := a.Cartesian(b);

function operator**<T>(a: sequence of T; n: integer): sequence of array of T; extensionmethod
  := a.CartesianPower(n);

begin
  var cart := (1..5) * ('a'..'h');
  cart.Println;
  var cart2 := (1..5) ** 3;
  cart2.Println;
end.