begin
  var (a,b) := ReadReal2('Введите a,b:');
  var min: real;
  if a<b then
    min := a
  else min := b;
  Print(min);
end.