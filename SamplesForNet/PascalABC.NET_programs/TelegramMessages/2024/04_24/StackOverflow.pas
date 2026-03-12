procedure p(n: integer);
begin
  var r1,r2,r3: real; // 40
  if n < 0 then
    exit;
  p(n-1)
end;

begin
  var n := 11000;
  p(n);
end.