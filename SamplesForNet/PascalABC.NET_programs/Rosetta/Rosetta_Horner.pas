// https://rosettacode.org/wiki/Horner's_rule_for_polynomial_evaluation#PascalABC.NET

function Horner(coeffs: array of real; x: real): real;
begin
  Result := 0;
  foreach var coeff in coeffs.Reverse do
    Result := Result * x + coeff
end;

begin
  Print(Horner(|-19.0, 7, -4, 6|, 3))
end.