unit MyUnit;

/// Сумма кавадратов a и b
function SumSquares(a,b: real) := a*a + b*b;

/// Вывод красным
procedure PrintRed(o: object) := 
  Print(#65534+o.ToString);

end.