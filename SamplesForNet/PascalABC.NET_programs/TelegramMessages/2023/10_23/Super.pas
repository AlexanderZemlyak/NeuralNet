function Superposition<T,T1,T2>(f: T1 -> T2; g: T -> T1): T -> T2 := x -> f(g(x));

function f(x: real) := x*x;

begin
  var fg := Superposition(f,Sin); 
  var gf := Superposition(Sin,f);
  Println(fg(2));
  Println(gf(2));
end.