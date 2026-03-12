procedure SumProd(a,b: real; var sum,prod: real);
begin
  sum := a + b;
  prod := a * b
end;

procedure SumProdShort(a,b: real; var sum,prod: real) := (sum,prod) := (a + b, a * b);

function SumProdF(a,b: real): (real,real);
begin
  Result := (a + b, a * b);
end;

function SumProdFShort(a,b: real) := (a + b, a * b);

begin
  var (a,b) := ReadReal2;
  var s,p: real;
  SumProd(a,b,s,p);
  Println(s,p);
  (s,p) := SumProdF(a,b);
  Println(s,p);
end.