// https://rosettacode.org/wiki/Arithmetic/Integer#PascalABC.NET

begin
  var a := 0;
  repeat
    a += 1;
    Print(a);
  until a mod 6 = 0;
end.